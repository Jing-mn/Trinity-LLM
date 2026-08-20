# tests/test_splitter.py
# 测试 src/rag_engine/splitter.py 的分块功能。

from langchain_core.documents import Document

from src.rag_engine.splitter import split_documents


def _make_long_document() -> Document:
    """构造一个足够长的文档，确保能被切成多个块（约 17 * 30 = 510 字符）。"""
    content = "这是第一句用于测试分块功能的句子。" * 30
    return Document(
        page_content=content,
        metadata={"source": "test.txt", "file_name": "test.txt"},
    )


def test_split_normal():
    """测试正常分块：chunk_size=100，overlap=20，长文档应产生多个块。"""
    doc = _make_long_document()
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1


def test_split_empty_list():
    """测试空文档列表：应返回空列表，不报错。"""
    assert split_documents([]) == []


def test_chunk_size_not_exceeded():
    """测试分块后每个块的长度不超过 chunk_size（RecursiveCharacterTextSplitter 的硬保证）。"""
    doc = _make_long_document()
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)

    for chunk in chunks:
        assert len(chunk.page_content) <= 100


def test_metadata_inherited():
    """测试分块后元数据被继承：每个小块都应带原文档的 source 和 file_name。"""
    doc = _make_long_document()
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata.get("source") == "test.txt"
        assert chunk.metadata.get("file_name") == "test.txt"
