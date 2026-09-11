import os
import pandas as pd
from sklearn.model_selection import train_test_split


def process_sst2(raw_dir="data/raw", save_dir="data/splits", val_size=0.1, random_seed=42):
    train_path = os.path.join(raw_dir, "sst2_train.tsv")
    dev_path = os.path.join(raw_dir, "sst2_dev.tsv")

    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        print("SST-2 files missing. Please run download_final.py first.")
        return

    train_df = pd.read_csv(train_path, sep='\t', header=None, names=['sentence', 'label'])
    dev_df = pd.read_csv(dev_path, sep='\t', header=None, names=['sentence', 'label'])

    # 去除可能混入的重复表头行（原始文件首行可能是 "sentence\tlabel"）
    for _df in (train_df, dev_df):
        _df.drop(_df.index[_df['label'].astype(str) == 'label'], inplace=True)
        _df.reset_index(drop=True, inplace=True)
    train_df['label'] = train_df['label'].astype(int)
    dev_df['label'] = dev_df['label'].astype(int)

    # 不再使用 stratify，避免类别样本过少问题
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_seed
    )
    test_df = dev_df

    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, "sst2_train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "sst2_val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "sst2_test.csv"), index=False)
    print(f"SST-2: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

if __name__ == "__main__":
    process_sst2()
    print("Done! SST-2 splits saved to data/splits/")