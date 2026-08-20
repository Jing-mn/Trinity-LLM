# src/rag_engine/rag_chain.py

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# 【国内网络修复】必须在导入 langchain_openai 之前设置 HF_ENDPOINT，
# 因为 langchain_openai 会间接导入 huggingface_hub，而后者在导入时只读取一次该环境变量。
# 否则后面加载 embedding 模型时会连 huggingface.co 触发 SSL 证书错误。
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.common.config import config
from src.rag_engine.vector_store import load_vector_store

logger = logging.getLogger(__name__)

# 【RAG 提示词】要求模型只依据检索到的上下文回答，避免胡说八道
RAG_PROMPT_TEMPLATE = """你是一个知识库问答助手。请仅根据下面提供的上下文回答用户的问题，不要编造上下文之外的信息。
如果上下文没有足够信息回答该问题，请直接说明"根据已有文档，无法回答该问题"。

上下文：
{context}

问题：{question}

回答："""


def _create_llm(temperature: float = 0.2) -> ChatOpenAI:
    """
    创建 OpenAI 兼容格式的 LLM（DeepSeek / 通义 / OpenAI 都走同一个接口）。
    """
    # 【关键】模型名在此显式写死，不读取 .env / 环境变量，
    # 避免被 LLM_MODEL_NAME（当前是 qwen-plus）覆盖导致 400 错误。
    # 该 DeepSeek 端点支持的模型名只有 deepseek-v4-pro 和 deepseek-v4-flash。
    return ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=temperature,
    )


def _format_docs(docs: List[Document]) -> str:
    """把检索到的文档块拼成一段文本，喂给 LLM，并带上来源标签方便溯源。"""
    return "\n\n".join(
        f"[来源: {doc.metadata.get('file_name', '未知')}]\n{doc.page_content}"
        for doc in docs
    )


def create_rag_chain(
    top_k: int = 4,
    temperature: float = 0.2,
    persist_directory: str = "./data/chroma_db",
) -> Tuple[Any, Any]:
    """
    创建 RAG 问答链。

    返回:
        (chain, retriever):
            chain     —— 接收一个问题，内部自动「检索 -> 拼提示词 -> 调 LLM -> 返回答案字符串」
            retriever —— 单独暴露出来，方便调用方取出引用来源做溯源
    """
    # 1. 加载已构建好的向量库
    vectorstore = load_vector_store(persist_directory)

    # 2. 包装成检索器（top_k 决定每次取几个最相关的文档块）
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 3. 组装链（用 LCEL 管道，不依赖 langchain.chains，避免老版本兼容问题）
    llm = _create_llm(temperature=temperature)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info(f"RAG 问答链构建完成（top_k={top_k}, temperature={temperature}）")
    return chain, retriever


def query_with_sources(
    question: str,
    chain,
    retriever,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    执行一次带溯源的 RAG 查询。

    参数:
        question: 用户问题
        chain: RAG 问答链
        retriever: 检索器（与 chain 内部共享同一个实例）
        top_k: 可选，动态调整本次检索的文档块数量；None 表示沿用创建链时的默认值

    返回:
        {"answer": str, "sources": List[str]} —— answer 是回答，sources 是引用来源文件名列表
    """
    # 支持动态调整检索数量：retriever 与 chain 内部共用同一实例，
    # 修改 search_kwargs 后，chain.invoke 内部的检索也会同步生效。
    # 查询结束后恢复原来的 k 值，避免影响后续未指定 top_k 的调用（防止状态泄漏）。
    original_k = retriever.search_kwargs.get("k")
    if top_k is not None and top_k > 0:
        retriever.search_kwargs["k"] = top_k

    try:
        # 先用检索器取回相关文档，提取来源（用于下面的引用溯源展示）
        docs = retriever.invoke(question)
        sources: List[str] = []
        for doc in docs:
            src = doc.metadata.get("file_name") or doc.metadata.get("source", "未知来源")
            if src not in sources:
                sources.append(src)

        # 再调用链生成回答
        answer = chain.invoke(question)

        return {"answer": answer, "sources": sources}
    finally:
        if original_k is not None:
            retriever.search_kwargs["k"] = original_k
