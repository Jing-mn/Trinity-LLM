# scripts/test_splitter.py

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 【Windows 控制台编码修复】中文 Windows 默认用 GBK 编码输出，
# 无法打印 emoji（✅⚠️ 等），会导致 UnicodeEncodeError，这里强制改为 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.rag_engine.loader import load_documents_from_folder
from src.rag_engine.splitter import split_documents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    docs_folder = "data/raw_docs"
    
    print("="*50)
    print("第1步：加载原始文档...")
    print("="*50)
    raw_docs = load_documents_from_folder(docs_folder)
    
    if not raw_docs:
        print("⚠️ 没有加载到任何文档，请检查 data/raw_docs 目录")
        sys.exit(1)
    
    print(f"\n原始文档块数量: {len(raw_docs)}")
    # 打印第1块的长度
    if raw_docs:
        print(f"第1块原始长度: {len(raw_docs[0].page_content)} 个字符")
        print(f"第1块预览: {raw_docs[0].page_content[:100].replace(chr(10), ' ')}...")
    
    print("\n" + "="*50)
    print("第2步：执行智能分块...")
    print("="*50)
    
    # 调用分块器，每块500字，重叠50字
    chunked_docs = split_documents(raw_docs, chunk_size=500, chunk_overlap=50)
    
    print(f"\n✅ 分块完成！")
    print(f"切分后文档块数量: {len(chunked_docs)}")
    
    if chunked_docs:
        print(f"第1块切分后长度: {len(chunked_docs[0].page_content)} 个字符")
        print(f"第1块内容预览: {chunked_docs[0].page_content[:100].replace(chr(10), ' ')}...")
        
        # 如果块数大于1，展示第2块的开头，让你看到"重叠"的效果
        if len(chunked_docs) > 1:
            print(f"\n第2块开头预览: {chunked_docs[1].page_content[:100].replace(chr(10), ' ')}...")
            
    print("\n" + "="*50)
    print("💡 提示：如果块数变多了，说明切分成功！接下来这些小块会被存入向量库。")