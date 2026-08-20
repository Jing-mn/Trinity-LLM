# Trinity-LLM

RAG + SFT + Agent 三位一体的大模型应用开发项目。当前重点实现并完善 RAG（检索增强生成）知识库问答链路，SFT（微调）与 Agent（智能体）方向规划中。

## 功能特性

- 文档加载：支持 PDF、Word（DOCX）、纯文本（TXT）三种格式
- 智能文本分块：按中文语义边界（段落、句号、逗号等）递归切分，支持自定义块大小与重叠
- 向量化存储：使用 Sentence-Transformers + Chroma 将文档转为向量并持久化
- RAG 问答：检索增强生成，答案基于知识库上下文，并支持动态调整检索数量（Top-K）
- 引用溯源：每个回答附带来源文件名，方便追溯信息出处
- CLI 命令行交互：一键构建知识库、命令行提问
- Gradio Web 界面：浏览器对话，内置"重建知识库"按钮
- FastAPI RESTful API：HTTP 接口 + 自动生成 Swagger 文档

> 规划中（未实现）：SFT 微调流水线（src/sft_trainer/）、Agent 智能体工作流（src/agent_workflows/）。

## 环境要求

- Python 3.10+
- 内存 4GB+（推荐 8GB，embedding 模型需要加载到内存）
- 大模型 API Key（DeepSeek 或任何 OpenAI 兼容接口）
- 首次运行需联网下载 embedding 模型（约 33MB，自动走国内镜像 hf-mirror.com）

## 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/Trinity-LLM.git
cd Trinity-LLM

# 2. 创建虚拟环境（Windows）
python -m venv .venv
.venv\Scripts\activate

# 3. 安装依赖
# 注意：默认 pip 源（清华镜像）缺少 langchain-huggingface 包，建议使用官方 PyPI 源
pip install -r requirements.txt -i https://pypi.org/simple

# 4. 配置环境变量
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```ini
# 大模型 API（DeepSeek / OpenAI 兼容接口）
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.deepseek.com
```

> 说明：当前 LLM 模型名在 `src/rag_engine/rag_chain.py` 中硬编码为 `deepseek-v4-flash`，
> `.env` 中的 `LLM_MODEL_NAME` 暂不生效，可忽略。

## 快速开始

将你的文档放入 `data/raw_docs/` 文件夹（支持 PDF/DOCX/TXT），然后按以下任意一种方式使用。

### 方式一：CLI 命令行问答

```bash
# 第一步：构建知识库（向量化存储）
python scripts/build_vectordb.py

# 第二步：命令行提问
python scripts/query_rag.py "这份文档的主要内容是什么？"
# 不带参数运行时使用默认问题
```

### 方式二：Gradio Web 界面

```bash
python src/web_ui/app.py
```

启动后自动打开浏览器访问 `http://127.0.0.1:7860`，直接在对话框提问即可。
界面底部提供"重建知识库"按钮，可重新加载 `data/raw_docs` 中的文档。

### 方式三：FastAPI 服务 + Swagger 文档

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

启动后：

- Swagger 交互式文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

## API 文档

### POST /rag/query

知识库问答接口，返回回答与引用来源。

**请求体**：

| 字段       | 类型    | 必填 | 说明                           |
| ---------- | ------- | ---- | ------------------------------ |
| `question` | string  | 是   | 用户问题                       |
| `top_k`    | integer | 否   | 检索文档块数量，默认 4，范围 1-10 |

```json
{
  "question": "这份文档的主要内容是什么？",
  "top_k": 4
}
```

**响应体**：

| 字段      | 类型             | 说明                           |
| --------- | ---------------- | ------------------------------ |
| `answer`  | string           | 基于知识库生成的回答           |
| `sources` | array of string  | 引用来源的文件名列表，可为空   |

```json
{
  "answer": "根据已有文档，这份文档的内容是……",
  "sources": ["test.txt"]
}
```

**示例 curl 命令**：

```bash
# Linux / macOS
curl -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "这份文档的主要内容是什么？", "top_k": 4}'

# Windows PowerShell（注意引号转义）
curl.exe -X POST http://127.0.0.1:8000/rag/query `
  -H "Content-Type: application/json" `
  -d '{\"question\": \"这份文档的主要内容是什么？\", \"top_k\": 4}'
```

## 项目结构

```
Trinity-LLM/
├── src/
│   ├── api/                      # FastAPI 服务
│   │   ├── main.py               # 应用入口，启动时预加载 RAG 系统
│   │   ├── models.py             # Pydantic 请求/响应模型
│   │   └── routes/
│   │       └── rag_router.py     # POST /rag/query 问答接口
│   ├── common/
│   │   └── config.py             # 全局配置（读取 .env）
│   ├── rag_engine/               # RAG 核心引擎
│   │   ├── loader.py             # 文档加载（PDF/DOCX/TXT）
│   │   ├── splitter.py           # 智能分块（中文语义分隔符）
│   │   ├── vector_store.py       # Chroma 向量库（构建/加载）
│   │   └── rag_chain.py          # RAG 问答链 + 引用溯源
│   └── web_ui/
│       └── app.py                # Gradio Web 界面（含重建知识库）
├── scripts/                      # CLI 脚本
│   ├── build_vectordb.py         # 构建知识库
│   ├── query_rag.py              # 命令行问答
│   ├── test_loader.py            # 加载功能自测
│   └── test_splitter.py          # 分块功能自测
├── data/
│   ├── raw_docs/                 # 放置原始文档（PDF/DOCX/TXT）
│   ├── chroma_db/                # 向量数据库（运行时生成）
│   └── sft_datasets/             # SFT 数据集（规划中）
├── configs/                      # YAML 配置文件（规划中）
├── notebooks/                    # 探索性 notebook（规划中）
├── tests/                        # 单元测试（规划中）
├── .env.example                  # 环境变量模板
├── .env                          # 环境变量（含 API Key，已被 .gitignore 忽略）
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## 技术栈

- **LangChain 0.3.x**：文档处理、文本分块、RAG 问答管道（LCEL）
- **Chroma 0.5.x**：向量数据库，持久化存储与相似度检索
- **Sentence-Transformers + BAAI/bge-small-zh-v1.5**：中文 embedding 模型
- **DeepSeek**（OpenAI 兼容接口）：大模型推理，通过 langchain-openai 调用
- **FastAPI + Uvicorn**：RESTful API 服务
- **Gradio**：Web 交互界面
- **Pydantic**：API 数据校验

## 引用溯源说明

系统通过以下机制追溯信息的来源：

1. **加载时打标签**：`loader.py` 读取每个文档后，为每个文档块写入 `file_name` 与 `source` 元数据，标明其来源文件。
2. **分块继承**：`splitter.py` 切分文本时，子块自动继承原始文档的元数据。
3. **存储保留**：向量化时，元数据与向量一同存入 Chroma。
4. **检索提取**：问答时，`rag_chain.py` 先从向量库检索相关文档块，提取其 `file_name` 作为引用来源，再让 LLM 基于这些上下文生成回答。

因此，每个回答都能返回"这段信息来自哪个文件"，实现可验证、可追溯的知识库问答。

## 许可证

MIT License
