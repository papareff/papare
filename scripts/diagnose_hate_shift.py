"""诊断 hate 的 val/test 落差：近重复泄漏 vs 真实分布偏移。
现象：train 99.05% / val 78.48% / test 52.80%（低于多数类基线 57.79%）
val 与 test 正类占比几乎相同（42.74% vs 42.21%），但准确率差 25.68pp。

两个竞争假设：
H1 val 泄漏 —— train 与 val 存在近重复（此前只清了精确重复），val 准确率虚高，
             test 52.80% 才是真实泛化水平 → hate 不适合做泛化任务
H2 分布偏移 —— train/val 与 test 文本特征系统性不同（表情、@提及、URL、大小写等），
             模型学到 train/val 的表层特征，在 test 上失效 → 需特征对齐或换任务
判据：比较 (train↔val) 与 (train↔test) 的近重复率；若前者显著高 → H1；
     若两者相当但特征分布差异大 → H2。
"""
import os
import re
import json
import hashlib
from collections import Counter
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
print("=" * 100)
print("hate val/test 落差诊断：近重复泄漏 vs 分布偏移")
print("=" * 100)

tr = pd.read_csv("data/raw/hate_train.csv")
va = pd.read_csv("data/raw/hate_val.csv")
te = pd.read_csv("data/raw/hate_test.csv")


def norm(s):
    """归一化：小写、去 @/#/URL、压缩空白，用于近重复判定"""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ngrams(s, n=3):
    t = s.split()
    return {" ".join(t[i:i+n]) for i in range(max(0, len(t) - n + 1))}


for name, df in [("train", tr), ("val", va), ("test", te)]:
    df["_norm"] = df["text"].map(norm)

print("\n【H1】近重复泄漏检验")
tr_norm = set(tr["_norm"])
tr_pref = set(tr["_norm"].str[:40])          # 前 40 字符前缀
tr_ng = {}
for s in tr["_norm"]:
    for g in ngrams(s, 4):
        tr_ng.setdefault(g, 0)
        tr_ng[g] += 1
tr_ng_set = set(tr_ng)

rows = []
for name, df in [("val", va), ("test", te)]:
    n = len(df)
    exact = int(df["_norm"].isin(tr_norm).sum())
    pref = int(df["_norm"].str[:40].isin(tr_pref).sum())
    # 4-gram 重合率：样本的 4-gram 有多少比例在 train 中出现过
    ov = []
    for s in df["_norm"]:
        g = ngrams(s, 4)
        ov.append(len(g & tr_ng_set) / len(g) if g else 0.0)
    ov = np.array(ov)
    hi = int((ov >= 0.8).sum())
    rows.append({"split": name, "n": n, "exact_dup": exact,
                 "exact_pct": exact / n * 100, "prefix40": pref,
                 "prefix_pct": pref / n * 100, "ng4_overlap_mean": float(ov.mean()),
                 "ng4_ge80pct": hi, "ng4_hi_pct": hi / n * 100})
    print(f"  {name:<5} n={n:>5,}  精确重复 {exact:>4} ({exact/n*100:>5.2f}%)  "
          f"前缀40字符 {pref:>4} ({pref/n*100:>5.2f}%)")
    print(f"        4-gram 平均重合率 {ov.mean()*100:>5.2f}%   "
          f"重合≥80% 的样本 {hi:>4} ({hi/n*100:>5.2f}%)")

v, t = rows[0], rows[1]
gap_exact = v["exact_pct"] - t["exact_pct"]
gap_ng = v["ng4_overlap_mean"] - t["ng4_overlap_mean"]
print(f"\n  val−test 精确重复率差   {gap_exact:+.2f}pp")
print(f"  val−test 4-gram重合率差 {gap_ng*100:+.2f}pp")
h1 = abs(gap_exact) > 5 or abs(gap_ng) > 0.05
print(f"  → H1 近重复泄漏: {'成立' if h1 else '不成立'}"
      f"（val 与 test 的泄漏程度{'显著不同' if h1 else '相当'}）")

print("\n【H2】文本特征分布偏移检验")
FEATS = {
    "@提及数": lambda s: len(re.findall(r"@\w+", str(s))),
    "URL数": lambda s: len(re.findall(r"http\S+|www\.\S+", str(s))),
    "话题#数": lambda s: len(re.findall(r"#\w+", str(s))),
    "非ASCII字符数": lambda s: sum(1 for c in str(s) if ord(c) > 127),
    "表情/符号数": lambda s: len(re.findall(r"[^\w\s@#]", str(s))),
    "大写字母占比": lambda s: (sum(1 for c in str(s) if c.isupper()) /
                            max(1, sum(1 for c in str(s) if c.isalpha()))),
    "词数": lambda s: len(str(s).split()),
    "字符数": lambda s: len(str(s)),
    "感叹/问号数": lambda s: len(re.findall(r"[!?]", str(s))),
}
feat_rows = []
for fname, fn in FEATS.items():
    a, b, c = (tr["text"].map(fn), va["text"].map(fn), te["text"].map(fn))
    # 标准化差异（以 train 的 std 为尺度）
    sd = a.std() if a.std() > 1e-9 else 1.0
    d_vt = (b.mean() - c.mean()) / sd
    feat_rows.append({"feature": fname, "train_mean": float(a.mean()),
                      "val_mean": float(b.mean()), "test_mean": float(c.mean()),
                      "val_minus_test_std": float(d_vt)})
    print(f"  {fname:<14} train {a.mean():>8.3f}   val {b.mean():>8.3f}   "
          f"test {c.mean():>8.3f}   (val−test)/σ_train {d_vt:>+7.3f}")
big = [r for r in feat_rows if abs(r["val_minus_test_std"]) > 0.2]
print(f"\n  标准化差异 >0.2σ 的特征: {[r['feature'] for r in big] or '无'}")
h2 = len(big) > 0
print(f"  → H2 分布偏移: {'成立' if h2 else '不成立'}")

print("\n【补充】标签噪声与难度线索")
for name, df in [("train", tr), ("val", va), ("test", te)]:
    dup_text = df[df.duplicated("text", keep=False)]
    if len(dup_text):
        conf = dup_text.groupby("text")["label"].nunique()
        print(f"  {name:<5} 组内标签矛盾的重复文本: {int((conf > 1).sum())} 组")
    else:
        print(f"  {name:<5} 无重复文本")
# 训练集内标签矛盾的归一化文本
g = tr.groupby("_norm")["label"].nunique()
print(f"  train 归一化后标签矛盾的文本: {int((g > 1).sum())} 组 "
      f"（占归一化唯一文本 {(g>1).sum()/len(g)*100:.2f}%）")

concl = []
if h1:
    concl.append("val 存在近重复泄漏，78.48% 虚高，test 52.80% 才反映真实泛化")
if h2:
    concl.append("train/val 与 test 存在特征分布偏移，模型学到表层特征")
if not h1 and not h2:
    concl.append("既无近重复泄漏也无显著特征偏移 → 属模型容量/过拟合问题，"
                 "应减少 epoch 或加正则，而非判定数据不可用")
concl.append(f"test 52.80% 低于多数类基线 57.79% → 当前 hate 裁决者不可用于泛化实验")

print("\n" + "=" * 100)
print("【结论】")
for i, c in enumerate(concl, 1):
    print(f"  {i}. {c}")
print("=" * 100)

out = {
    "leakage": rows, "h1_leakage": bool(h1),
    "gap_exact_pct": float(gap_exact), "gap_ng4": float(gap_ng),
    "features": feat_rows, "h2_shift": bool(h2),
    "big_shift_features": [r["feature"] for r in big],
    "conclusions": concl,
    "metrics": {"train_acc_pct": 99.05, "val_acc_pct": 78.48,
                "test_acc_pct": 52.80, "test_majority_baseline_pct": 57.79,
                "best_checkpoint_epoch": 5, "trained_epochs": 8},
}
os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/hate_shift_diagnosis.json", "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("结果已写入 experiments/results/hate_shift_diagnosis.json")
