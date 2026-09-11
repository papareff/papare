import os
import requests
import pandas as pd
from io import BytesIO
import zipfile


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"✓ {dest} already exists, skipping.")
        return
    print(f"⬇ Downloading {url} ...")
    headers = {"User-Agent": "Mozilla/5.0"}  # 防止某些网站拒绝
    try:
        response = requests.get(url, verify=False, timeout=120, headers=headers)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(response.content)
        print(f"✓ Saved to {dest}")
    except Exception as e:
        print(f"✗ Failed to download {dest}: {e}")


def download_sst2_from_glue():
    """从 GLUE 官方源下载 SST-2 压缩包并提取 train.tsv 和 dev.tsv"""
    url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    zip_path = "data/raw/SST-2.zip"
    extract_dir = "data/raw"

    if os.path.exists("data/raw/train.tsv") and os.path.exists("data/raw/dev.tsv"):
        print("✓ SST-2 files already exist, skipping.")
        return

    print("⬇ Downloading SST-2 from GLUE official source...")
    try:
        response = requests.get(url, verify=False, timeout=120)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        # 重命名文件（解压后是 SST-2/ 子目录）
        if os.path.exists("data/raw/SST-2/train.tsv"):
            os.rename("data/raw/SST-2/train.tsv", "data/raw/sst2_train.tsv")
            os.rename("data/raw/SST-2/dev.tsv", "data/raw/sst2_dev.tsv")
            # 删除多余文件夹
            import shutil
            shutil.rmtree("data/raw/SST-2")
        os.remove(zip_path)
        print("✓ SST-2 downloaded and extracted.")
    except Exception as e:
        print(f"✗ Failed to download SST-2: {e}")


def download_ag_news():
    """从 Hugging Face 镜像下载 AG News CSV（直接可用）"""
    files = [
        ("ag_news_train.csv", "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/train.csv"),
        ("ag_news_test.csv", "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/test.csv"),
    ]
    for fname, url in files:
        dest = os.path.join("data/raw", fname)
        download_file(url, dest)


def main():
    os.makedirs("data/raw", exist_ok=True)
    print("=" * 50)
    print("Starting dataset download...")
    print("=" * 50)
    download_sst2_from_glue()
    download_ag_news()
    print("\n" + "=" * 50)
    print("All downloads completed!")
    print("Now you can run: python scripts/prepare_data.py")
    print("=" * 50)


if __name__ == "__main__":
    main()