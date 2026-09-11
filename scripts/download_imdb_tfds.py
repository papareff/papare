import os
import pandas as pd
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

def save_imdb():
    # 加载 IMDb 数据集（自动下载，缓存到 ~/tensorflow_datasets）
    print("正在下载 IMDb 数据集（首次会较慢，请耐心等待）...")
    train_ds, test_ds = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        as_supervised=True,
        shuffle_files=False
    )

    # 转换为 Pandas DataFrame
    def ds_to_df(ds):
        texts = []
        labels = []
        for text, label in tfds.as_numpy(ds):
            texts.append(text.decode('utf-8'))
            labels.append(label)  # label 是 0/1，无需转换
        return pd.DataFrame({'text': texts, 'label': labels})

    train_df = ds_to_df(train_ds)
    test_df = ds_to_df(test_ds)

    # 划分验证集
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['label']
    )

    os.makedirs('data/raw', exist_ok=True)
    train_df.to_csv('data/raw/imdb_train.csv', index=False)
    val_df.to_csv('data/raw/imdb_val.csv', index=False)
    test_df.to_csv('data/raw/imdb_test.csv', index=False)

    print(f"IMDb saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

if __name__ == "__main__":
    save_imdb()