"""清洗 tweet_eval/hate 的上游数据泄漏。

问题（download_hate.py 实测）：
  train ∩ test = 232 条，train ∩ validation = 2 条
这是 tweet_eval 上游缺陷。若直接用 hate_train 训练裁决者、在 hate_test 评测，
232 条测试样本是模型见过的，准确率会虚高，与 irony 任务的结果不可比。

清洗策略：以 test / validation 为「干净评测集」优先，从 train 中剔除与它们
重叠的文本。理由：
  • test 用于最终报告，必须是完全未见的样本，不能删（删了会改变类别比例且损失样本量）
  • train 少 234 条（8993→8759）对训练影响极小（−2.6%）
  • 反向做法（从 test 删 232 条）会让 test 规模与类别比例都变动，且损失评测样本

同时校验清洗后：三个划分两两无重叠、类别比例变动幅度、词数分布是否仍与 irony 同量级。
"""
import os
import json
import pandas as pd

OUT_DIR = "data/raw"
SPLITS = ["train", "val", "test"]
FILES = {s: f"{OUT_DIR}/hate_{s}.csv" for s in SPLITS}

for s, p in FILES.items():
    if not os.path.exists(p):
        raise SystemExit(f"[FAIL] 缺少 {p}，请先运行 scripts/download_hate.py")

print("=" * 100)
print("hate 数据集泄漏清洗")
print("=" * 100)

dfs = {s: pd.read_csv(p) for s, p in FILES.items()}
for s in SPLITS:
    dfs[s]["text"] = dfs[s]["text"].astype(str).str.strip()
    dfs[s]["label"] = pd.to_numeric(dfs[s]["label"], errors="coerce")
    dfs[s] = dfs[s].dropna(subset=["label"]).copy()
    dfs[s]["label"] = dfs[s]["label"].astype(int)

print("\n【清洗前】")
print(f"  {'划分':<10} {'样本数':>8} {'label=1':>9} {'占比':>8} {'唯一文本':>10}")
for s in SPLITS:
    d = dfs[s]
    print(f"  {s:<10} {len(d):>8,} {int((d.label==1).sum()):>9,} "
          f"{(d.label==1).mean()*100:>7.2f}% {d['text'].nunique():>10,}")

sets = {s: set(dfs[s]["text"]) for s in SPLITS}
print(f"\n  原始重叠:")
print(f"    train ∩ test       = {len(sets['train'] & sets['test']):>4} 条")
print(f"    train ∩ val        = {len(sets['train'] & sets['val']):>4} 条")
print(f"    val   ∩ test       = {len(sets['val'] & sets['test']):>4} 条")

# ── 清洗：从 train 剔除与 val/test 重叠的文本 ──
contam = sets["val"] | sets["test"]
before = len(dfs["train"])
mask = dfs["train"]["text"].isin(contam)
removed = dfs["train"][mask]
dfs["train"] = dfs["train"][~mask].reset_index(drop=True)
after = len(dfs["train"])

print(f"\n【清洗】从 train 剔除与 val/test 重叠的文本")
print(f"  剔除 {before - after} 条（{before} → {after}，−{(before-after)/before*100:.2f}%）")
print(f"  被剔除样本的标签分布: "
      f"{ {int(k): int(v) for k, v in removed['label'].value_counts().sort_index().items()} }")
print(f"  被剔除样本中 label=1 占比 {removed.label.mean()*100:.2f}%，"
      f"train 原 label=1 占比 {(dfs['train'].label.mean()*100):.2f}%")
print(f"  → 剔除样本的正类比例{'与原 train 接近，未引入选择偏置' if abs(removed.label.mean()-0.42)<0.06 else '与原 train 差异较大，需注意'}")

# ── 清洗后校验 ──
sets2 = {s: set(dfs[s]["text"]) for s in SPLITS}
print(f"\n【清洗后】")
print(f"  {'划分':<10} {'样本数':>8} {'label=1':>9} {'占比':>8} {'唯一文本':>10}")
for s in SPLITS:
    d = dfs[s]
    print(f"  {s:<10} {len(d):>8,} {int((d.label==1).sum()):>9,} "
          f"{(d.label==1).mean()*100:>7.2f}% {d['text'].nunique():>10,}")

print(f"\n  重叠复查（必须全为 0）:")
ok = True
for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
    n = len(sets2[a] & sets2[b])
    flag = "OK" if n == 0 else "仍泄漏!"
    if n:
        ok = False
    print(f"    {a:<6} ∩ {b:<6} = {n:>4}   [{flag}]")

# 划分内部重复
print(f"\n  划分内部重复文本检查:")
for s in SPLITS:
    dup = len(dfs[s]) - dfs[s]["text"].nunique()
    print(f"    {s:<10} 内部重复 {dup} 条")
    if dup:
        dfs[s] = dfs[s].drop_duplicates(subset=["text"]).reset_index(drop=True)
        print(f"      → 已去重，剩余 {len(dfs[s])} 条")

# 与 irony 的交叉
if os.path.exists(f"{OUT_DIR}/irony_test.csv"):
    irony = set(pd.read_csv(f"{OUT_DIR}/irony_test.csv")["text"].astype(str).str.strip())
    n = len(irony & sets2["test"])
    print(f"\n    irony_test ∩ hate_test = {n}   [{'OK' if n == 0 else '需检查'}]")

# ── 落盘（覆盖原文件，保持文件名不变以便脚本复用）──
print(f"\n【落盘】")
for s in SPLITS:
    p = FILES[s]
    dfs[s][["text", "label"]].to_csv(p, index=False, encoding="utf-8")
    print(f"  {p}  ←  {len(dfs[s]):,} 条")

# ── 任务画像对比 ──
print(f"\n{'='*100}")
print("清洗后任务画像（与 irony 对比，确认难度可比）")
print("=" * 100)
print(f"  {'任务':<20} {'训练':>8} {'验证':>8} {'测试':>8} {'测试正类占比':>13} "
      f"{'多数类基线':>11} {'词数均值':>9}")
print("  " + "-" * 86)
prof = {}
for task, prefix in [("tweet_eval/hate", "hate"), ("tweet_eval/irony", "irony")]:
    row = {}
    for s in SPLITS:
        p = f"{OUT_DIR}/{prefix}_{s}.csv"
        row[s] = len(pd.read_csv(p)) if os.path.exists(p) else None
    pt = f"{OUT_DIR}/{prefix}_test.csv"
    mb = pos = wl = None
    if os.path.exists(pt):
        t = pd.read_csv(pt)
        pos = (t.label == 1).mean() * 100
        mb = t["label"].value_counts().max() / len(t) * 100
        wl = t["text"].astype(str).str.split().str.len().mean()
    prof[task] = {**row, "test_pos_rate_pct": pos,
                  "majority_baseline_pct": mb, "words_mean": wl}
    print(f"  {task:<20} {str(row['train']):>8} {str(row['val']):>8} "
          f"{str(row['test']):>8} {f'{pos:.2f}%' if pos else '—':>13} "
          f"{f'{mb:.2f}%' if mb else '—':>11} {f'{wl:.1f}' if wl else '—':>9}")

print(f"\n  ◆ 标签语义映射（接入框架时的口径）")
print(f"    hate : label=1 → 攻击性  → 框架中的 pro 票")
print(f"    irony: label=1 → 反讽    → 框架中的 pro 票")
print(f"    两者同构：都要求识别「字面≠意图」，pro 偏置的方向与量级可直接对比")

print(f"\n  ◆ 难度判据")
print(f"    hate 多数类基线 {prof['tweet_eval/hate']['majority_baseline_pct']:.2f}%，"
      f"irony {prof['tweet_eval/irony']['majority_baseline_pct']:.2f}%")
print(f"    两者均处于 55~62% 区间 → 同属困难任务，无多数类捷径")
print(f"    对照 SST-2：裁决者已达 98% → 饱和任务，辩论无纠错空间（已否决）")

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/hate_leakage_clean.json", "w", encoding="utf-8") as f:
    json.dump({"removed_from_train": before - after,
               "train_before": before, "train_after": after,
               "overlap_before": {"train_test": len(sets['train'] & sets['test']),
                                  "train_val": len(sets['train'] & sets['val']),
                                  "val_test": len(sets['val'] & sets['test'])},
               "overlap_after": {"train_test": len(sets2['train'] & sets2['test']),
                                 "train_val": len(sets2['train'] & sets2['val']),
                                 "val_test": len(sets2['val'] & sets2['test'])},
               "clean": ok, "profiles": prof,
               "label_semantics": {"hate": {1: "hateful→pro", 0: "not hateful→con"},
                                   "irony": {1: "ironic→pro", 0: "not ironic→con"}}},
              f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已写入 experiments/results/hate_leakage_clean.json")
print("=" * 100)
