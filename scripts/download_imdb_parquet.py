import os
import requests

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping.")
        return
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {dest} ({total_size} bytes)")

def main():
    os.makedirs("data/raw", exist_ok=True)
    # 使用镜像站加速
    base_url = "https://hf-mirror.com/datasets/imdb/resolve/main"
    files = [
        ("train.parquet", f"{base_url}/train.parquet"),
        ("test.parquet", f"{base_url}/test.parquet"),
    ]
    for fname, url in files:
        download_file(url, os.path.join("data/raw", fname))

    print("\nDownload complete. Now converting to CSV...")
    import pandas as pd
    train_df = pd.read_parquet("data/raw/train.parquet")
    test_df = pd.read_parquet("data/raw/test.parquet")
    train_df.to_csv("data/raw/imdb_train.csv", index=False)
    test_df.to_csv("data/raw/imdb_test.csv", index=False)
    print(f"IMDb CSV saved: train={len(train_df)}, test={len(test_df)}")

if __name__ == "__main__":
    main()