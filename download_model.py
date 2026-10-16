import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
# os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"   # 如果 SSL 还报错，再把这一行打开

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-small-zh-v1.5",
    local_dir="./models/bge-small-zh-v1.5",
    local_dir_use_symlinks=False
)

print("✅ 模型下载完成")