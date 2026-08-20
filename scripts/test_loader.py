# scripts/test_loader.py

import sys
import os
import logging

# 【为什么要这样写？】因为脚本在 scripts 文件夹里，要调用 src 里的代码，
# 必须把项目根目录（TRINITY_LLM）加入系统的搜索路径中，否则 Python 会报错”找不到模块”。
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 【Windows 控制台编码修复】中文 Windows 默认用 GBK 编码输出，
# 无法打印 emoji（✅⚠️ 等），会导致 UnicodeEncodeError，这里强制改为 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.rag_engine.loader import load_documents_from_folder

# 配置日志打印到终端，颜色无所谓，主要是能看到信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # 指定我们的文档存放路径
    docs_folder = "data/raw_docs"
    
    print(f"开始测试加载文件夹: {docs_folder}")
    
    # 调用我们刚才写好的加载函数
    documents = load_documents_from_folder(docs_folder)
    
    # 打印结果
    print("\n" + "="*50)
    print(f"✅ 测试结果：共加载了 {len(documents)} 个文档块")
    print("="*50)
    
    # 如果加载到了数据，展示前 2 块的内容预览
    if documents:
        for i, doc in enumerate(documents[:2]):
            print(f"\n--- 文档块 {i+1} 的来源: {doc.metadata.get('file_name', '未知')} ---")
            # 只打印前 100 个字符，防止内容太长刷屏
            content_preview = doc.page_content[:100].replace("\n", " ")
            print(f"内容预览: {content_preview}...")
    else:
        print("⚠️ 警告：没有加载到任何文档块！请检查 data/raw_docs 里是否有 PDF、Word 或 TXT 文件。")