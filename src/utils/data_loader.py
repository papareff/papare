import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple

def load_and_split_dataset(
    dataset_name: str,
    save_dir: str = "data/splits",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    从 HuggingFace 加载数据集，划分训练/验证/测试集，保存为 CSV。
    支持 sst2 (独立版) 和 ag_news。
    """
    if dataset_name == "sst2":
        # 独立 sst2 数据集：train + validation
        dataset = load_dataset("sst2")
        train_df = dataset["train"].to_pandas()
        val_df = dataset["validation"].to_pandas()
        # 将 validation 作为测试集（或者进一步拆分）
        # 这里我们保留 validation 作为测试集，因为大小适中
        # 同时从 train 中再划分 10% 作为验证集
        train_df, new_val_df = train_test_split(
            train_df, test_size=val_size, random_state=random_seed, stratify=train_df["label"]
        )
        # 合并原来的 val_df 作为测试集（但为了统一，我们将 validation 作为测试集）
        # 更标准做法：使用 validation 作为测试，但为了有验证集，我们从 train 切分。
        # 这里我们使用 validation 作为测试集，并从 train 切分验证集。
        return {
            "train": train_df,
            "val": new_val_df,
            "test": val_df
        }
    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
        train_df = dataset["train"].to_pandas()
        test_df = dataset["test"].to_pandas()
        train_df, val_df = train_test_split(
            train_df, test_size=val_size, random_state=random_seed, stratify=train_df["label"]
        )
        return {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def save_splits(splits: Dict[str, pd.DataFrame], dataset_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for split_name, df in splits.items():
        file_path = os.path.join(save_dir, f"{dataset_name}_{split_name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {split_name} set: {len(df)} samples -> {file_path}")

def get_dataset_stats(splits: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    return {name: len(df) for name, df in splits.items()}