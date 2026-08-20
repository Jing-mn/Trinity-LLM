# tests/test_loader.py
# 测试 src/rag_engine/loader.py 的文档加载功能。
# 全部使用 pytest 的 tmp_path 临时目录，不污染真实的 data/ 目录。

import pytest

from src.rag_engine.loader import load_documents_from_folder


def test_load_txt_file(tmp_path):
    """测试加载单个 txt 文件：应返回 1 个文档，内容与元数据正确。"""
    # 在临时目录里写一个 UTF-8 编码的 txt 文件
    txt_file = tmp_path / "hello.txt"
    txt_file.write_text("你好，这是一个用于测试的文档。", encoding="utf-8")

    docs = load_documents_from_folder(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].page_content == "你好，这是一个用于测试的文档。"
    # loader 会给每个文档块打上 file_name 和 source 元数据
    assert docs[0].metadata["file_name"] == "hello.txt"
    assert "source" in docs[0].metadata


def test_load_missing_folder_raises(tmp_path):
    """测试加载不存在的文件夹：应抛出 FileNotFoundError。"""
    missing_dir = tmp_path / "not_exists"
    with pytest.raises(FileNotFoundError):
        load_documents_from_folder(str(missing_dir))


def test_load_empty_folder_returns_empty(tmp_path):
    """测试加载空文件夹：应返回空列表。"""
    docs = load_documents_from_folder(str(tmp_path))
    assert docs == []


def test_skip_unsupported_format(tmp_path):
    """测试加载不支持的文件格式：应跳过该文件，不影响整体加载结果。"""
    # 同时放入一个受支持的 txt 和一个不支持的 png
    (tmp_path / "ok.txt").write_text("支持的文档内容", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    docs = load_documents_from_folder(str(tmp_path))

    # 只有 txt 被加载，png 被跳过
    assert len(docs) == 1
    assert docs[0].metadata["file_name"] == "ok.txt"


def test_all_unsupported_returns_empty(tmp_path):
    """测试文件夹里只有不支持格式的文件：应返回空列表（不报错）。"""
    (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    docs = load_documents_from_folder(str(tmp_path))
    assert docs == []
