import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def save_agnews():
    dataset = load_dataset("ag_news")
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # 划分验证集 (10%)
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['label']
    )

    os.makedirs("data/raw", exist_ok=True)
    train_df.to_csv("data/raw/agnews_train.csv", index=False)
    val_df.to_csv("data/raw/agnews_val.csv", index=False)
    test_df.to_csv("data/raw/agnews_test.csv", index=False)
    print(f"AG News saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    save_agnews()