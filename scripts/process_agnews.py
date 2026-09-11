import os
import pandas as pd
from sklearn.model_selection import train_test_split


def process_agnews():
    # 使用已有的文件
    train_file = "data/raw/ag_news_train.csv"
    test_file = "data/raw/ag_news_test.csv"

    # 检查文件是否存在
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("文件不存在，请检查路径。")
        return

    # 读取 CSV（AG News 无 header，列：label, title, description）
    train_df = pd.read_csv(train_file, header=None, names=['label', 'title', 'description'])
    test_df = pd.read_csv(test_file, header=None, names=['label', 'title', 'description'])

    # 合并标题和描述为文本
    train_df['text'] = train_df['title'] + " " + train_df['description']
    test_df['text'] = test_df['title'] + " " + test_df['description']

    # 标签从 1 开始，转换为 0 开始
    train_df['label'] = train_df['label'] - 1
    test_df['label'] = test_df['label'] - 1

    # 从训练集划分验证集（10%）
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['label']
    )

    # 只保留 text 和 label
    train_df = train_df[['text', 'label']]
    val_df = val_df[['text', 'label']]
    test_df = test_df[['text', 'label']]

    # 保存
    os.makedirs("data/raw", exist_ok=True)
    train_df.to_csv("data/raw/agnews_train.csv", index=False)
    val_df.to_csv("data/raw/agnews_val.csv", index=False)
    test_df.to_csv("data/raw/agnews_test.csv", index=False)

    print(f"AG News processed: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    process_agnews()