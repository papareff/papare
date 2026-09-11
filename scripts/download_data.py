import os
import requests
from tqdm import tqdm

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping.")
        return
    # 使用流式下载，显示进度条
    response = requests.get(url, stream=True, verify=False)  # 关闭SSL验证有时能解决
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Downloaded {dest}")

# 镜像站替换
base_url = "https://hf-mirror.com"
files = [
    ("sst2_train.tsv", "/datasets/sst2/resolve/main/data/train.tsv"),
    ("sst2_dev.tsv", "/datasets/sst2/resolve/main/data/dev.tsv"),
    ("ag_news_train.csv", "/datasets/ag_news/resolve/main/data/train.csv"),
    ("ag_news_test.csv", "/datasets/ag_news/resolve/main/data/test.csv"),
]

os.makedirs("data/raw", exist_ok=True)
for filename, path in files:
    url = base_url + path
    dest = os.path.join("data/raw", filename)
    print(f"Downloading {url} -> {dest}")
    try:
        download_file(url, dest)
    except Exception as e:
        print(f"Failed to download {filename}: {e}")