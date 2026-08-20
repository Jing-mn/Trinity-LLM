# scripts/build_vectordb.py

import sys
import os
import logging

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 【Windows 控制台编码修复】中文 Windows 默认用 GBK 编码输出，
# 无法打印 emoji（🚀✅🎉 等），会导致 UnicodeEncodeError，这里强制改为 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.rag_engine.loader import load_documents_from_folder
from src.rag_engine.splitter import split_documents
from src.rag_engine.vector_store import build_vector_store

# 配置日志（打印详细过程）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("开始执行 RAG 知识库构建流水线")
    print("🚀 " * 20 + "\n")
    
    # ---- 1. 配置参数（你可以在这里调整） ----
    DOCS_FOLDER = "data/raw_docs"        # 你的原始文档放哪
    CHUNK_SIZE = 500                     # 分块大小
    CHUNK_OVERLAP = 50                   # 分块重叠
    PERSIST_DIR = "./data/chroma_db"     # 向量库存哪
    
    # ---- 2. 加载 ----
    print("[1/3] 正在加载原始文档...")
    raw_docs = load_documents_from_folder(DOCS_FOLDER)
    if not raw_docs:
        print("❌ 错误：没有找到任何文档，请检查 data/raw_docs 文件夹！")
        sys.exit(1)
    print(f"✅ 加载完成，共 {len(raw_docs)} 个原始文档块")
    
    # ---- 3. 分块 ----
    print("\n[2/3] 正在智能分块...")
    chunks = split_documents(raw_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"✅ 分块完成，共生成 {len(chunks)} 个文档块")
    
    # ---- 4. 向量化存储 ----
    print("\n[3/3] 正在向量化并存入 Chroma（这步可能需要 1-3 分钟，请耐心等待）...")
    vectorstore = build_vector_store(chunks, persist_directory=PERSIST_DIR)
    
    # ---- 5. 完成 ----
    print("\n" + "🎉 " * 20)
    print("恭喜！知识库构建完成！")
    print(f"🎉 向量库已保存在: {PERSIST_DIR}")
    print("🎉 " * 20 + "\n")
    print("💡 提示：你随时可以再次运行此脚本，新增的文档会自动追加（需额外配置，目前覆盖重建）")