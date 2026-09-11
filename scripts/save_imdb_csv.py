import os
from datasets import load_dataset
import pandas as pd


def save_imdb():
    print("Loading IMDb dataset from Hugging Face...")
    dataset = load_dataset("imdb")

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    os.makedirs("data/raw", exist_ok=True)
    train_df.to_csv("data/raw/imdb_train.csv", index=False)
    test_df.to_csv("data/raw/imdb_test.csv", index=False)
    print(f"IMDb saved: train={len(train_df)}, test={len(test_df)}")


if __name__ == "__main__":
    save_imdb()