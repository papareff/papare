import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
from sklearn.model_selection import train_test_split

# AG News 数据集在 Hugging Face 上的 CSV 文件链接（通过镜像站）
base_url = "https://huggingface.co/datasets/ag_news/resolve/main/data"


# 注意：实际文件路径可能稍有不同，根据官方数据集结构调整
# 也可以直接使用 huggingface hub 上的原始 CSV 文件

def download_from_url(url, dest):
    import requests
    print(f"Downloading {url} ...")
    response = requests.get(url)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        f.write(response.content)
    print(f"Saved to {dest}")


def save_agnews():
    os.makedirs("data/raw", exist_ok=True)
    train_url = f"{base_url}/train.csv"
    test_url = f"{base_url}/test.csv"

    # 下载 CSV 文件
    download_from_url(train_url, "data/raw/agnews_train_raw.csv")
    download_from_url(test_url, "data/raw/agnews_test_raw.csv")

    # 读取并处理
    train_df = pd.read_csv("data/raw/agnews_train_raw.csv")
    test_df = pd.read_csv("data/raw/agnews_test_raw.csv")

    # 列名：通常是 'class', 'title', 'description'
    # 合并 title 和 description 为 text
    train_df['text'] = train_df['title'] + " " + train_df['description']
    test_df['text'] = test_df['title'] + " " + test_df['description']
    train_df = train_df[['class', 'text']].rename(columns={'class': 'label'})
    test_df = test_df[['class', 'text']].rename(columns={'class': 'label'})

    # 划分验证集 (10%)
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['label']
    )

    train_df.to_csv("data/raw/agnews_train.csv", index=False)
    val_df.to_csv("data/raw/agnews_val.csv", index=False)
    test_df.to_csv("data/raw/agnews_test.csv", index=False)
    print(f"AG News saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    save_agnews()