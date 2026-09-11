"""逐分片下载 Qwen2-7B-Instruct 权重（更稳健：串行 + 自动重试 + 断点续传）。

先下载小文件，再逐个串行下载 4 个权重分片，每个分片失败自动重试。
走 hf-mirror.com 镜像，禁用 xet 协议（镜像不支持，会报 401）。
"""
import os, time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = r"E:\huggingface_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Qwen/Qwen2-7B-Instruct"

files = list_repo_files(REPO_ID)
# 大权重分片放最后串行下载
weight_files = [f for f in files if f.endswith(".safetensors")]
other_files = [f for f in files if not f.endswith(".safetensors")]
ordered = other_files + weight_files

print(f"共 {len(ordered)} 个文件，其中权重分片 {len(weight_files)} 个")

for i, f in enumerate(ordered):
    for attempt in range(1, 6):
        try:
            p = hf_hub_download(repo_id=REPO_ID, filename=f)
            size_mb = os.path.getsize(p) / 1024**2
            print(f"[{i+1}/{len(ordered)}] 完成: {f} ({size_mb:.1f} MB)")
            break
        except Exception as e:
            print(f"[{i+1}/{len(ordered)}] {f} 第 {attempt} 次失败: {type(e).__name__}: {str(e)[:120]}")
            if attempt == 5:
                raise
            time.sleep(3 * attempt)

print("全部文件下载完成")
