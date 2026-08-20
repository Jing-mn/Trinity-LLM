# src/rag_engine/splitter.py

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    将长文档切成适合 AI 处理的小块。
    
    参数:
        documents: 从 loader 加载的原始文档列表
        chunk_size: 每块最多多少字符（建议 300-800，中文环境下推荐500）
        chunk_overlap: 块与块之间重叠多少字符（保持上下文连贯性）
    
    返回:
        切分后的小文档块列表
    """
    if not documents:
        logger.warning("传入的文档列表为空，无需切分")
        return []
    
    logger.info(f"开始切分 {len(documents)} 个原始文档块...")
    
    # 【核心武器】RecursiveCharacterTextSplitter 是 LangChain 最常用的分块器
    # 它会按优先级从高到低尝试用分隔符切分，保证语义完整性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 【重要】中文环境下，分隔符优先级：段落 > 句号/感叹号/问号 > 逗号 > 空格
        separators=[
            "\n\n",    # 双换行（段落边界）
            "\n",      # 单换行
            "。",      # 中文句号
            "！",      # 感叹号
            "？",      # 问号
            "；",      # 分号
            "，",      # 逗号
            " ",       # 空格
            ""         # 终极手段：按字符硬切（保证不超过 chunk_size）
        ],
        length_function=len,  # 用字符数计算长度（中文比较准）
    )
    
    # 执行切分
    split_docs = text_splitter.split_documents(documents)
    
    # 【优化小技巧】切分后，把原始文档的元数据继承给每个小块
    # 这样每个小块都知道自己来自哪个文件，方便追溯
    for i, doc in enumerate(split_docs):
        # 如果原文档有 source 元数据，确保每个子块都带上
        if "source" not in doc.metadata and documents:
            # 尝试从第一个原始文档继承来源（适用于简单场景）
            pass  # 实际上 split_documents 会自动继承 metadata，这里只是保险
    
    logger.info(f"切分完成！从 {len(documents)} 块扩展为 {len(split_docs)} 个小块")
    return split_docs