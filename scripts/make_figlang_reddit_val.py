"""为 FIGLANG reddit 从 train 切 10% 作 val（官方无 val 划分），供训练早停与选优。
twitter 已有官方 val（500 条），reddit 缺失。
切分用与 prepare_figlang_data.py 一致的 seed=42，分层抽样保持正类占比。
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

src = "data/raw/figlang_reddit_train.csv"
df = pd.read_csv(src)
n0 = len(df)
pos0 = df["label"].mean()
print(f"原 train: {n0:,} 条  正类占比 {pos0*100:.2f}%")

tr, va = train_test_split(df, test_size=0.1, random_state=42,
                          stratify=df["label"])
tr = tr.reset_index(drop=True)
va = va.reset_index(drop=True)

# 不覆盖原始 train，写到新文件名以保留完整原始数据
tr.to_csv("data/raw/figlang_reddit_train_split.csv", index=False, encoding="utf-8")
va.to_csv("data/raw/figlang_reddit_val.csv", index=False, encoding="utf-8")

print(f"切分后 train: {len(tr):,} 条  正类 {tr['label'].mean()*100:.2f}% "
      f"→ figlang_reddit_train_split.csv（原 figlang_reddit_train.csv 保留不动）")
print(f"新建 val    : {len(va):,} 条  正类 {va['label'].mean()*100:.2f}%")

# 泄漏校验：train 与 val 文本不得重叠
ov = len(set(tr["text"]) & set(va["text"]))
print(f"train∩val 文本重叠: {ov} 条  {'✓ 无泄漏' if ov == 0 else '✗ 有泄漏'}")

# 与 test 的泄漏校验
te = pd.read_csv("data/raw/figlang_reddit_test.csv")
ov_te = len(set(tr["text"]) & set(te["text"])) + len(set(va["text"]) & set(te["text"]))
print(f"(train∪val)∩test 文本重叠: {ov_te} 条  {'✓ 无泄漏' if ov_te == 0 else '⚠ 有重叠'}")
