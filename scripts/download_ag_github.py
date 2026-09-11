import os
import requests

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping.")
        return
    print(f"Downloading {url} ...")
    try:
        r = requests.get(url, verify=False, timeout=120)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(r.content)
        print(f"Saved to {dest}")
    except Exception as e:
        print(f"Failed: {e}")

os.makedirs("data/raw", exist_ok=True)
download_file(
    "https://raw.githubusercontent.com/mhjabreel/CharCNN/master/data/ag_news_csv/train.csv",
    "data/raw/ag_news_train.csv"
)
download_file(
    "https://raw.githubusercontent.com/mhjabreel/CharCNN/master/data/ag_news_csv/test.csv",
    "data/raw/ag_news_test.csv"
)