"""下载 SemEval-2018 Task 3 反讽检测数据集（Task A，二分类：反讽/非反讽）。
优先走 HuggingFace datasets 库（经 hf-mirror.com 镜像），输出统一的 text/label CSV 划分。
label: 1 = ironic（反讽），0 = not ironic（非反讽）。
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("data/raw", exist_ok=True)

def load_irony():
    """尝试多种来源加载反讽数据，返回 (train_df, test_df)，均含 text/label 列。"""
    from datasets import load_dataset

    # 候选：tweet_eval（cardiffnlp 官方）的 irony 子集 = SemEval-2018 Task 3A
    candidates = [
        ("cardiffnlp/tweet_eval", "irony"),
    ]
    last_err = None
    for name, cfg in candidates:
        try:
            print(f"尝试加载: {name} ({cfg})")
            ds = load_dataset(name, cfg) if cfg else load_dataset(name)
            print("  加载成功，子集:", list(ds.keys()))
            return normalize(ds)
        except Exception as e:
            last_err = e
            print(f"  失败: {type(e).__name__}: {e}")
    raise RuntimeError(f"所有数据源均失败，最后错误: {last_err}")


def normalize(ds):
    """把不同来源的 dataset 统一为 (train_df, test_df)，列：text, label(0/1)。"""
    keys = list(ds.keys())
    # 选取训练与测试子集
    train_key = next((k for k in keys if 'train' in k), keys[0])
    test_key = next((k for k in keys if k in ('test', 'gold_test', 'Test')), None)
    if test_key is None:
        test_key = next((k for k in keys if 'test' in k), None)

    train_ds = ds[train_key].to_pandas()
    test_ds = ds[test_key].to_pandas() if test_key else None
    print(f"  训练子集 {train_key}: {len(train_ds)} 行, 列={list(train_ds.columns)}")
    if test_ds is not None:
        print(f"  测试子集 {test_key}: {len(test_ds)} 行, 列={list(test_ds.columns)}")
    return coerce(train_ds), (coerce(test_ds) if test_ds is not None else None)


def coerce(df):
    """把 dataframe 规整为 text/label。label: 1=ironic, 0=not。"""
    df = df.copy()
    # 文本列
    text_col = next((c for c in ['tweet_text', 'text', 'Tweet text', 'tweet'] if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"找不到文本列，现有列: {list(df.columns)}")
    # 标签列
    label_col = next((c for c in ['label', 'Label', 'irony', 'SemEval2018-T3-train.txt'] if c in df.columns), None)
    if label_col is None:
        # 有些 SemEval 封装把标签列命名特殊，取最后一个数值列
        numeric_cols = [c for c in df.columns if c != text_col and pd.api.types.is_numeric_dtype(df[c])]
        label_col = numeric_cols[-1] if numeric_cols else None
    if label_col is None:
        raise ValueError(f"找不到标签列，现有列: {list(df.columns)}")

    out = pd.DataFrame()
    out['text'] = df[text_col].astype(str).str.strip()
    out['label'] = pd.to_numeric(df[label_col], errors='coerce')
    out = out.dropna(subset=['text', 'label'])
    out['label'] = out['label'].astype(int)
    # 去除空文本
    out = out[out['text'].str.len() > 0].reset_index(drop=True)
    return out


def main():
    train_df, test_df = load_irony()
    print(f"\n原始: train={len(train_df)}, test={0 if test_df is None else len(test_df)}")
    print("train 标签分布:\n", train_df['label'].value_counts())

    if test_df is None or len(test_df) == 0:
        # 没有独立测试集时从训练集切出
        train_df, test_df = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    # 从训练集再切验证集
    tr, val = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    tr.to_csv("data/raw/irony_train.csv", index=False)
    val.to_csv("data/raw/irony_val.csv", index=False)
    test_df.to_csv("data/raw/irony_test.csv", index=False)
    print(f"\n保存: train={len(tr)}, val={len(val)}, test={len(test_df)}")
    print("test 标签分布:\n", test_df['label'].value_counts())
    print("\n样例:")
    print(train_df.head(3).to_string())


if __name__ == "__main__":
    main()
