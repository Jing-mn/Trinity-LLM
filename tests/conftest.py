# tests/conftest.py
# pytest 运行前的公共配置（会被 pytest 自动加载）

import os
import sys

# 将项目根目录加入 sys.path，确保各测试文件能 `import src.rag_engine.*`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 【国内网络修复】在导入任何 src 模块之前设置 HuggingFace 镜像，
# 否则加载 embedding 模型时会连 huggingface.co 触发 SSL 证书错误。
# 必须在第一个 import langchain_huggingface 之前生效（huggingface_hub 只读一次 HF_ENDPOINT）。
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
