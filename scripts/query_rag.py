# scripts/query_rag.py

import sys
import os
import logging

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 【Windows 控制台编码修复】中文 Windows 默认用 GBK 编码输出，
# 无法打印中文以外的字符，会导致乱码或 UnicodeEncodeError，这里强制改为 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from src.rag_engine.rag_chain import create_rag_chain, query_with_sources

# 加载 .env 文件中的环境变量（API Key 等）
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # 检查是否从命令行传入了问题，如果没有就用默认问题
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
    else:
        user_question = "请简要介绍这份文档的主要内容是什么？"
    
    print("正在初始化 RAG 系统（加载向量库和模型）...")
    
    # 创建问答链（Top-K = 4，温度设为 0.2 让回答更严谨）
    chain, retriever = create_rag_chain(top_k=4, temperature=0.2)
    
    print("初始化完成。")
    print("=" * 60)
    print(f"问题: {user_question}")
    print("=" * 60)
    
    # 执行查询
    result = query_with_sources(user_question, chain, retriever)
    
    # 打印回答
    print("\n回答:")
    print(result["answer"])
    
    # 打印引用来源（这就是引用溯源功能）
    print("\n" + "=" * 60)
    if result["sources"]:
        print("引用来源:")
        for source in result["sources"]:
            print(f"  - {source}")
    else:
        print("未引用任何具体文档（可能未检索到相关内容）。")
    
    print("=" * 60)