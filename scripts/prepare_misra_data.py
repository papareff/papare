"""准备两阶段训练数据：Misra 28K 新闻标题讽刺集切分 + 域内诊断集。

阶段1（域适应预训练）：Misra train 21281 / val 2660 / test 2661（实际落盘行数）
阶段2（任务微调）    ：SemEval irony_train 2575 / irony_val 287（已有，不动）
主评测               ：SemEval irony_test 784（已有，不动）
跨域诊断             ：Misra test 2671（量化域偏移影响）

输出列名统一为 text,label，与现有 irony_*.csv 完全一致，训练脚本可直接复用。
"""
import os
import io
import hashlib
import pandas as pd
import pyarrow.parquet as pq

SRC = "data/tmp_probe/Heschmat__news-headlines-dataset-sarcasm-detection_train-00000-of-00001.parquet.parquet"
OUT_DIR = "data/raw"
SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(SRC):
    raise SystemExit(f"找不到已缓存的 Misra 数据集: {SRC}\n请先运行 scripts/probe_parquet_rows.py 下载")

with open(SRC, "rb") as f:
    raw = pq.read_table(io.BytesIO(f.read())).to_pandas()

print("=" * 92)
print("阶段一：加载与清洗 Misra 28K")
print("=" * 92)
print(f"原始行数: {len(raw):,}")

# 列名标准化：headline→text, is_sarcastic→label
df = raw.rename(columns={"headline": "text", "is_sarcastic": "label"})
df["text"] = df["text"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="coerce")

# 清洗：去空、去纯符号、去重
n0 = len(df)
df = df.dropna(subset=["text", "label"])
df = df[df["text"].str.len() > 3]
df = df[df["text"].str.contains(r"[A-Za-z]", regex=True)]
df["label"] = df["label"].astype(int)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"清洗后:   {len(df):,}（去除 {n0 - len(df):,} 条空值/重复/无效）")

vc = df["label"].value_counts().sort_index()
for k, v in vc.items():
    print(f"  label={k}: {v:,} ({v/len(df)*100:.1f}%)")

print(f"\n阶段二：按 {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{1-TRAIN_FRAC-VAL_FRAC:.0%} 切分（seed={SEED}）")
shuf = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n = len(shuf)
n_tr = int(n * TRAIN_FRAC)
n_va = int(n * VAL_FRAC)
train = shuf.iloc[:n_tr].reset_index(drop=True)
val = shuf.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
test = shuf.iloc[n_tr + n_va:].reset_index(drop=True)

paths = {}
for name, part in [("train", train), ("val", val), ("test", test)]:
    p = f"{OUT_DIR}/misra_{name}.csv"
    part[["text", "label"]].to_csv(p, index=False, encoding="utf-8")
    paths[name] = p
    n1 = int((part["label"] == 1).sum())
    print(f"  misra_{name}.csv : {len(part):>6,} 条  讽刺 {n1:>5,} ({n1/len(part)*100:.1f}%)  → {p}")

# 泄漏检查：三个阶段之间不得有重叠文本
print(f"\n阶段三：数据泄漏检查")
tr_set = set(train["text"])
va_set = set(val["text"])
te_set = set(test["text"])
leaks = {
    "train∩val": len(tr_set & va_set),
    "train∩test": len(tr_set & te_set),
    "val∩test": len(va_set & te_set),
}
for k, v in leaks.items():
    flag = "OK" if v == 0 else "泄漏!"
    print(f"  {k:<14}: {v}   [{flag}]")

# 跨域检查：Misra 与 SemEval 是否有重叠（不同数据源，应为 0）
semeval = pd.read_csv("data/raw/irony_train.csv")
overlap = len(set(semeval["text"].astype(str).str.strip()) & tr_set)
print(f"  SemEval∩Misra : {overlap}   [{'OK' if overlap == 0 else '需检查'}]")

print(f"\n阶段四：数据集总览（两阶段训练用）")
print("-" * 92)
print(f"{'阶段':<24} {'数据源':<26} {'训练样本':>10} {'验证样本':>10} {'正类占比':>10}")
print("-" * 92)
se_val = pd.read_csv("data/raw/irony_val.csv")
se_test = pd.read_csv("data/raw/irony_test.csv")
print(f"{'①域适应预训练':<22} {'Misra 新闻标题':<24} {len(train):>10,} {len(val):>10,} "
      f"{(train.label==1).mean()*100:>9.1f}%")
print(f"{'②任务微调':<22} {'SemEval-2018 推特':<22} {len(semeval):>10,} {len(se_val):>10,} "
      f"{(semeval.label==1).mean()*100:>9.1f}%")
print(f"{'③主评测':<22} {'SemEval test':<24} {'—':>10} {len(se_test):>10,} "
      f"{(se_test.label==1).mean()*100:>9.1f}%")
print(f"{'④跨域诊断':<22} {'Misra test':<24} {'—':>10} {len(test):>10,} "
      f"{(test.label==1).mean()*100:>9.1f}%")
print("-" * 92)
print(f"\n参数量/样本数比变化:")
print(f"  原 DistilBERT + SemEval 2575      : {66_000_000/len(semeval):>8,.0f} : 1")
print(f"  阶段1 DistilBERT + Misra {len(train):,} : {66_000_000/len(train):>8,.0f} : 1")
print(f"  阶段1 RoBERTa-base + Misra {len(train):,}: {125_000_000/len(train):>8,.0f} : 1")
print(f"\nMisra 数据集指纹: md5={hashlib.md5(open(SRC,'rb').read()).hexdigest()[:16]}")
print("数据准备完成。")
