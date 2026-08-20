# src/web_ui/app.py

import importlib.util
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# 1. 路径设置：把项目根目录加入 sys.path，确保 `from src.xxx import ...` 能正常导入
#    本文件位于 src/web_ui/，向上三级即项目根目录
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 【国内网络修复】gradio 会间接导入 huggingface_hub，而后者在导入时只读取一次 HF_ENDPOINT。
# 必须在 import gradio 之前设置，否则后续加载 embedding 模型会连 huggingface.co 触发 SSL 错误。
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _ensure_dependencies():
    """启动前自动检查并安装缺失依赖（import 模块名 -> pip 包名）。"""
    requirements = {
        "sentence_transformers": "sentence-transformers",  # RAG 向量化所依赖的 embedding 库
        "gradio": "gradio",
    }
    missing = [pip_name for mod, pip_name in requirements.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"检测到缺失依赖：{', '.join(missing)}，正在自动安装...")
    for pip_name in missing:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        except subprocess.CalledProcessError as e:
            print(f"❌ 自动安装 {pip_name} 失败：{e}")
            print(f"   请手动执行：{sys.executable} -m pip install {pip_name}")
            sys.exit(1)
    print("✅ 依赖安装完成。")


# 2. 依赖检查（必须在 import gradio / sentence-transformers 之前执行）
_ensure_dependencies()

import gradio as gr

from src.rag_engine.loader import load_documents_from_folder
from src.rag_engine.splitter import split_documents
from src.rag_engine.rag_chain import create_rag_chain, query_with_sources
from src.rag_engine.vector_store import build_vector_store

# ---------------------------------------------------------------------------
# 3. 配置常量
# ---------------------------------------------------------------------------
DOCS_FOLDER = "data/raw_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PERSIST_DIR = "./data/chroma_db"
TOP_K = 4
# 注意：集合名要和 src/rag_engine/vector_store.py 里保持一致，重建时才能清理旧数据
COLLECTION_NAME = "trinity_rag_collection"

# 全局 RAG 状态（懒加载：真正用到时才初始化，导入本模块不会触发重量级操作）
chain = None
retriever = None


def _init_rag():
    """初始化（或重新初始化）RAG 链与检索器。"""
    global chain, retriever
    print("正在加载 RAG 系统...")
    chain, retriever = create_rag_chain(top_k=TOP_K, temperature=0.2)
    print("RAG 系统加载完成。")


def _delete_existing_collection():
    """删除旧向量库 collection，避免重建时新旧数据叠加。"""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=os.path.abspath(PERSIST_DIR))
        client.delete_collection(COLLECTION_NAME)
        print("已清理旧向量库 collection")
    except Exception as e:
        # 首次构建时旧库不存在，忽略即可
        print(f"无旧向量库需清理（{e}）")


def chat_response(message, history):
    """处理用户消息，返回带引用来源的回答文本。"""
    if not message or not message.strip():
        return "请输入有效的问题。"

    if chain is None or retriever is None:
        return "RAG 系统尚未就绪，请先运行 scripts/build_vectordb.py 构建知识库，或点击下方「重建知识库」按钮。"

    try:
        result = query_with_sources(question=message, chain=chain, retriever=retriever)
        answer = result["answer"]
        sources = result["sources"]

        response = answer
        if sources:
            response += "\n\n---\n**引用来源：**\n"
            for src in sources:
                response += f"- {src}\n"
        else:
            response += "\n\n---\n未引用具体文档。"
        return response
    except Exception as e:
        return f"处理问题时出错：{str(e)}"


def rebuild_knowledge_base():
    """重建知识库：重新加载 data/raw_docs 文档 -> 分块 -> 向量化 -> 重新加载 RAG 链。"""
    try:
        print("[重建知识库] 正在加载文档...")
        docs = load_documents_from_folder(DOCS_FOLDER)
        if not docs:
            return "❌ 未在 data/raw_docs 中找到任何文档（支持 PDF/DOCX/TXT），请先放入文档。"

        chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"[重建知识库] 分块完成，共 {len(chunks)} 块")

        _delete_existing_collection()

        print("[重建知识库] 正在向量化并写入 Chroma（约需 1-3 分钟）...")
        build_vector_store(chunks, persist_directory=PERSIST_DIR)

        print("[重建知识库] 重新加载 RAG 链...")
        _init_rag()

        return f"✅ 知识库重建完成，共 {len(chunks)} 个文档块，现在可以正常提问了。"
    except Exception as e:
        return f"❌ 重建失败：{e}"


# ---------------------------------------------------------------------------
# 4. 构建 Gradio 界面
# ---------------------------------------------------------------------------
def _respond(message, history):
    """Gradio Chatbot 回调：把 chat_response 的输出塞进对话历史。"""
    if not message or not message.strip():
        return "", history
    answer = chat_response(message, history)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", history


with gr.Blocks(title="Trinity-LLM RAG 知识库问答") as demo:
    gr.Markdown("# Trinity-LLM RAG 知识库问答")
    gr.Markdown("基于本地文档的知识库问答系统。在下方输入问题，系统会从已构建的向量库中检索相关信息并回答。")

    chatbot = gr.Chatbot(height=480, label="对话")
    msg = gr.Textbox(placeholder="请输入你的问题，按 Enter 发送", label="问题")

    with gr.Row():
        submit_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.ClearButton([msg, chatbot], value="清空对话")
        rebuild_btn = gr.Button("🔄 重建知识库", variant="secondary")

    rebuild_status = gr.Markdown()

    gr.Examples(
        examples=[
            "请简要介绍这份文档的主要内容。",
            "这份文档提到了哪些关键信息？",
            "帮我总结一下文档的核心观点。",
        ],
        inputs=msg,
    )

    msg.submit(_respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(_respond, [msg, chatbot], [msg, chatbot])
    rebuild_btn.click(rebuild_knowledge_base, outputs=rebuild_status)


if __name__ == "__main__":
    # 启动时初始化 RAG（失败不阻塞 UI，聊天/重建时会给提示）
    try:
        _init_rag()
    except Exception as e:
        print(f"⚠️ 初始化 RAG 失败：{e}")
        print("  界面仍会启动，可在页面下方点击「重建知识库」或先运行 scripts/build_vectordb.py。")

    # 5. 启动：端口 7860，自动打开浏览器
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,   # 启动后自动打开浏览器
        share=False,
        theme="soft",
    )
