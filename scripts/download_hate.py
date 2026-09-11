"""下载 tweet_eval 的 hate 子集，切分为与 irony 一致的 text,label 格式。

任务定位：隐性攻击检测（二分类，多数类基线 57.85%）。
与反讽的同构性：两者都要求识别「字面≠意图」——反讽是字面褒义实际贬义，
隐性仇恨是表层无攻击词但实际攻击。故视角分化辩论的机理可迁移。

⚠️ 难度需实测确认：57.85% 多数类基线不算高，但 hate 任务在 BERT 类模型上
   通常仅 60~70%，属困难区间；与 SST-2（裁决者 98%，饱和）性质不同。
"""
import urllib.request
import io
import os
import sys

import pyarrow.parquet as pq
import pandas as pd

BASE = "https://hf-mirror.com/datasets/tweet_eval/resolve/main/hate"
OUT_DIR = "data/raw"
SPLITS = {
    "train": ("train-00000-of-00001.parquet", "hate_train.csv"),
    "validation": ("validation-00000-of-00001.parquet", "hate_val.csv"),
    "test": ("test-00000-of-00001.parquet", "hate_test.csv"),
}

os.makedirs(OUT_DIR, exist_ok=True)


def fetch(url, timeout=240):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


print("=" * 100)
print("下载 tweet_eval / hate 子集")
print("=" * 100)

saved = {}
for split, (rel, out_name) in SPLITS.items():
    url = f"{BASE}/{rel}"
    try:
        raw = fetch(url)
    except Exception as e:
        print(f"  [FAIL] {split}: {type(e).__name__}: {e}")
        sys.exit(1)
    df = pq.read_table(io.BytesIO(raw)).to_pandas()
    df = df.rename(columns={"text": "text"})
    if "label" not in df.columns:
        print(f"  [FAIL] {split}: 缺少 label 列，实际列 {list(df.columns)}")
        sys.exit(1)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    out_path = f"{OUT_DIR}/{out_name}"
    df[["text", "label"]].to_csv(out_path, index=False, encoding="utf-8")
    vc = df["label"].value_counts().sort_index()
    saved[split] = out_path
    print(f"\n  {split:<11} {len(df):>6,} 条  → {out_path}")
    for k, v in vc.items():
        print(f"      label={k}: {v:>6,} ({v/len(df)*100:>5.1f}%)")
    print(f"      多数类基线 {vc.max()/len(df)*100:.2f}%")
    wl = df["text"].str.split().str.len()
    print(f"      词数: 均值 {wl.mean():.1f}  中位 {wl.median():.0f}  "
          f"p99 {wl.quantile(.99):.0f}  最大 {wl.max()}")
    print(f"      样例（label=1 攻击性）:")
    for t in df[df.label == 1]["text"].head(2):
        print(f"        • {t[:110]}")
    print(f"      样例（label=0 非攻击）:")
    for t in df[df.label == 0]["text"].head(2):
        print(f"        • {t[:110]}")

# ── 泄漏检查 ──
print(f"\n{'─'*100}")
print("泄漏检查（三个划分之间不得有重叠文本）")
print("─" * 100)
sets = {}
for split, p in saved.items():
    sets[split] = set(pd.read_csv(p)["text"].astype(str).str.strip())
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    n = len(sets[a] & sets[b])
    print(f"  {a:<11} ∩ {b:<11} = {n}   [{'OK' if n == 0 else '泄漏!'}]")

# 与现有反讽数据是否有重叠（不同数据集，应为 0）
if os.path.exists("data/raw/irony_test.csv"):
    irony = set(pd.read_csv("data/raw/irony_test.csv")["text"].astype(str).str.strip())
    n = len(irony & sets["test"])
    print(f"  irony_test  ∩ hate_test   = {n}   [{'OK' if n == 0 else '需检查'}]")

# ── 与反讽任务的对比 ──
print(f"\n{'─'*100}")
print("任务对比（确认 hate 属困难任务，不同于饱和的 SST-2）")
print("─" * 100)
print(f"  {'任务':<22} {'训练集':>9} {'验证集':>9} {'测试集':>9} {'多数类基线':>11}")
print("  " + "-" * 66)
rows = []
for task, prefix in [("tweet_eval/hate", "hate"), ("tweet_eval/irony", "irony")]:
    vals = {}
    for sp, suf in [("train", "train"), ("val", "val"), ("test", "test")]:
        p = f"{OUT_DIR}/{prefix}_{suf}.csv"
        if os.path.exists(p):
            vals[sp] = len(pd.read_csv(p))
        else:
            vals[sp] = None
    pt = f"{OUT_DIR}/{prefix}_test.csv"
    mb = None
    if os.path.exists(pt):
        t = pd.read_csv(pt)
        mb = t["label"].value_counts().max() / len(t) * 100
    rows.append({"task": task, **vals, "majority_baseline_pct": mb})
    print(f"  {task:<22} {str(vals['train']):>9} {str(vals['val']):>9} "
          f"{str(vals['test']):>9} {f'{mb:.2f}%' if mb else '—':>11}")
print(f"  {'（参照）SST-2 情感':<20} {'—':>9} {'—':>9} {'—':>9} "
      f"{'裁决者已达 98%':>11}  ← 饱和，辩论无纠错空间")

import json
os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/hate_dataset_profile.json", "w",
          encoding="utf-8") as f:
    json.dump({"splits": saved, "comparison": rows,
               "label_semantics": {"0": "not hateful", "1": "hateful"},
               "note": ("隐性攻击检测，与反讽同构（字面≠意图）；"
                        "label=1 为攻击性，对应框架中的 pro 票")},
              f, ensure_ascii=False, indent=2)
print(f"\n结果已写入 experiments/results/hate_dataset_profile.json")
print("=" * 100)
