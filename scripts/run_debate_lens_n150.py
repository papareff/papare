"""视角分化辩论正式复核（反讽任务，n=150，贪心解码可复现）。

一次完成三件事：
  1) n=150 视角分化辩论 + 纯裁决者基线（同批次配对）
  2) 门控阈值扫描（τ ∈ [0.50, 1.01]）
  3) 按裁决者置信度分带诊断（检验辩论增益的方向与显著性）

样本批次与历史 n=150 实验完全一致（SEED=42 分层抽样 + random.shuffle），
但结果为贪心解码，完全可复现。

历史 n=150 结果（irony_debate_neutral_with.csv）因"双方提示词逐字相同 →
98.7% 同票 → 判决退化为常数预测器"而作废，本脚本产出替代结果。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import time
import random
import numpy as np
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

NUM = 150
SEED = 42
ROUNDS = 1                      # 与验证阶段一致；rounds=2 另做敏感性分析
TEST_CSV = "data/raw/irony_test.csv"
OUT_DIR = "experiments/results"
LABEL_VOTE = {1: "pro", 0: "con"}

os.makedirs(OUT_DIR, exist_ok=True)

# ── 与历史实验完全相同的抽样批次 ──
with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["task"] = "irony"
config["debate"]["rounds"] = ROUNDS
config["debate"]["neutral"] = True

test_df = pd.read_csv(TEST_CSV)
test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce")
test_df = test_df.dropna(subset=["label", "text"])
test_df["label"] = test_df["label"].astype(int)

full_batch = (
    test_df.groupby("label", group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), NUM // 2), random_state=SEED))
)
records = full_batch.reset_index(drop=True).to_dict("records")
random.seed(SEED)
random.shuffle(records)
samples = records[:NUM]

n_pro_true = sum(1 for s in samples if int(s["label"]) == 1)
print(f"样本: {len(samples)} 条（反讽 {n_pro_true} / 非反讽 {len(samples)-n_pro_true}，"
      f"反讽占比 {n_pro_true/len(samples)*100:.1f}%）", flush=True)

debate_config = dict(config)
debate_config["judge"] = {
    "name": "./models/irony_distilbert", "device": "cuda", "max_length": 128
}
pipeline = InferencePipeline(debate_config, use_debate=True, retriever=None, num_shots=0)
print(f"辩论者 temperature={pipeline.pro_debater.temperature}（0.0=贪心解码，可复现）\n", flush=True)


def correct_of(true_label, pred):
    return pred == LABEL_VOTE[int(true_label)]


# ══════════ 1) 视角分化辩论（n=150）══════════
print(f"===== 视角分化辩论（n={len(samples)}，预计约 {len(samples)*30/60:.0f} 分钟）=====", flush=True)
rows, lats = [], []
t_start = time.time()
for i, s in enumerate(samples):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    r = pipeline.run(s)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    lats.append(dt)
    v = r.get("debate_votes", {})
    rows.append({
        "sample_id": i, "text": s["text"], "true_label": int(s["label"]),
        "gold": LABEL_VOTE[int(s["label"])],
        "predicted": r["decision"], "judge_vote": r.get("judge_vote"),
        "judge_confidence": r.get("judge_confidence"),
        "pro_vote": v.get("pro"), "con_vote": v.get("con"),
        "votes_agree": int(v.get("pro") == v.get("con")),
        "latency_sec": round(dt, 2),
        "correct": int(correct_of(s["label"], r["decision"])),
    })
    if (i + 1) % 10 == 0 or i == 0:
        el = time.time() - t_start
        eta = el / (i + 1) * (len(samples) - i - 1)
        run_acc = np.mean([x["correct"] for x in rows]) * 100
        print(f"  [{i+1}/{len(samples)}] 当前准确率 {run_acc:.2f}%  "
              f"已用 {el/60:.1f}min  预计剩余 {eta/60:.1f}min", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/debate_lens_n150_with.csv", index=False, encoding="utf-8-sig")

# ══════════ 2) 纯裁决者基线（同批次）══════════
print(f"\n===== 纯裁决者基线（n={len(samples)}）=====", flush=True)
base_rows = []
for i, s in enumerate(samples):
    res = pipeline.judge.make_decision(s, debate_votes=None)
    base_rows.append({
        "sample_id": i, "true_label": int(s["label"]), "gold": LABEL_VOTE[int(s["label"])],
        "predicted": res["decision"], "judge_confidence": res["judge_confidence"],
        "correct": int(correct_of(s["label"], res["decision"])),
    })
base_df = pd.DataFrame(base_rows)
base_df.to_csv(f"{OUT_DIR}/debate_lens_n150_judge_only.csv", index=False, encoding="utf-8-sig")

acc_debate = df["correct"].mean() * 100
acc_judge = base_df["correct"].mean() * 100
agree = df["votes_agree"].mean() * 100
pro_rate = (df["predicted"] == "pro").mean() * 100
true_pro = (df["true_label"] == 1).mean() * 100


def se(pct, n):
    return math.sqrt(pct * (100 - pct) / n)


def two_prop_se(p1, p2, n):
    """两相关比例差的标准误（保守用独立近似）"""
    return math.sqrt((p1 * (100 - p1) + p2 * (100 - p2)) / n)


print("\n" + "=" * 72)
print("复核结果（n=%d，贪心解码可复现）" % NUM)
print("=" * 72)
print("① 机制健康度")
print("   双方投票一致率 : %.1f%%  （历史坏版本 98.7%%）" % agree)
print("   判决 pro 占比  : %.1f%%  （真实反讽占比 %.1f%%，历史坏版本 98.7%%）" % (pro_rate, true_pro))
print("   常数预测器检验 : %s" % (
    "已脱离" if abs(pro_rate - true_pro) > 3.0 else "仍退化"))
print("")
print("② 准确率对照（含 95%% 置信区间）")
print("   视角分化辩论 : %.2f%%  ±%.2fpp" % (acc_debate, se(acc_debate, NUM) * 1.96))
print("   纯裁决者     : %.2f%%  ±%.2fpp" % (acc_judge, se(acc_judge, NUM) * 1.96))
delta = acc_debate - acc_judge
d_se = two_prop_se(acc_debate, acc_judge, NUM) * 1.96
print("   辩论增益     : %+.2fpp  ±%.2fpp" % (delta, d_se))
print("   显著性判定   : %s" % (
    "显著（置信区间不含 0）" if abs(delta) > d_se else "不显著（置信区间含 0）"))
print("")
print("③ 分类别命中")
for lab, name in [(1, "真反讽"), (0, "真非反讽")]:
    sub = df[df.true_label == lab]
    sb = base_df[base_df.true_label == lab]
    print("   %-8s n=%2d  辩论 %.2f%%   裁决者 %.2f%%   差 %+.2fpp" % (
        name, len(sub), sub["correct"].mean() * 100, sb["correct"].mean() * 100,
        (sub["correct"].mean() - sb["correct"].mean()) * 100))

# ══════════ 3) 门控阈值扫描 ══════════
print("\n④ 门控阈值扫描（conf ≥ τ 用裁决者，conf < τ 用辩论）")
print("   %-8s %10s %10s %12s %12s" % ("τ", "准确率", "触发率", "加权延迟", "vs纯裁决者"))
lat_debate = float(np.mean(lats))
gate_rows = []
best = None
for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]:
    low = df["judge_confidence"] < tau
    pred = np.where(low, df["predicted"], df["judge_vote"])
    acc = (pd.Series(pred, index=df.index) == df["gold"]).mean() * 100
    trig = low.mean() * 100
    wlat = trig / 100 * lat_debate + (1 - trig / 100) * 0.01
    gate_rows.append({"tau": tau, "accuracy": acc, "trigger_rate": trig,
                      "weighted_latency": wlat})
    print("   %-8.2f %9.2f%% %9.1f%% %11.2fs %+11.2fpp" % (
        tau, acc, trig, wlat, acc - acc_judge))
    if best is None or acc > best["accuracy"]:
        best = {"tau": tau, "accuracy": acc, "trigger_rate": trig, "weighted_latency": wlat}
gate_df = pd.DataFrame(gate_rows)
gate_df.to_csv(f"{OUT_DIR}/debate_lens_gate_scan.csv", index=False, encoding="utf-8-sig")
print("   ★ 最优 τ=%.2f: 准确率 %.2f%%（触发率 %.1f%%，延迟 %.2fs）" % (
    best["tau"], best["accuracy"], best["trigger_rate"], best["weighted_latency"]))
print("     vs 纯裁决者 %+.2fpp，vs 全程辩论 %+.2fpp，延迟为全程辩论的 %.0f%%" % (
    best["accuracy"] - acc_judge, best["accuracy"] - acc_debate,
    best["weighted_latency"] / lat_debate * 100))

# ══════════ 4) 分带诊断 ══════════
print("\n⑤ 分带诊断（辩论增益是否随置信度反转）")
merged = df.merge(base_df[["sample_id", "predicted", "correct"]].rename(
    columns={"predicted": "judge_pred", "correct": "judge_correct"}), on="sample_id")
BINS = [0.0, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]
BANDS = ["<0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.80", "≥0.80"]
band_rows = []
print("   %-12s %4s %10s %10s %10s" % ("置信带", "n", "裁决者", "辩论后", "增益"))
for lo, hi, name in zip(BINS[:-1], BINS[1:], BANDS):
    sub = merged[(merged.judge_confidence >= lo) & (merged.judge_confidence < hi)]
    if len(sub) == 0:
        continue
    aj = sub["judge_correct"].mean() * 100
    ad = sub["correct"].mean() * 100
    band_rows.append({"band": name, "n": len(sub), "judge_acc": round(aj, 2),
                      "debate_acc": round(ad, 2), "delta_pp": round(ad - aj, 2)})
    print("   %-12s %4d %9.2f%% %9.2f%% %+9.2fpp" % (name, len(sub), aj, ad, ad - aj))
pd.DataFrame(band_rows).to_csv(f"{OUT_DIR}/debate_lens_band_diagnosis.csv",
                              index=False, encoding="utf-8-sig")

# ══════════ 汇总落盘 ══════════
summary = pd.DataFrame([{
    "n": NUM, "rounds": ROUNDS, "decoding": "greedy",
    "acc_debate_pct": round(acc_debate, 2), "acc_judge_pct": round(acc_judge, 2),
    "delta_pp": round(delta, 2), "delta_ci95_pp": round(d_se, 2),
    "significant": bool(abs(delta) > d_se),
    "vote_agree_pct": round(agree, 2), "pred_pro_pct": round(pro_rate, 2),
    "true_pro_pct": round(true_pro, 2),
    "avg_latency_sec": round(lat_debate, 2),
    "best_tau": best["tau"], "best_gate_acc_pct": round(best["accuracy"], 2),
    "best_gate_trigger_pct": round(best["trigger_rate"], 2),
    "best_gate_latency_sec": round(best["weighted_latency"], 2),
}])
summary.to_csv(f"{OUT_DIR}/debate_lens_n150_summary.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 72)
print("结果文件：")
for f in ["debate_lens_n150_with.csv", "debate_lens_n150_judge_only.csv",
          "debate_lens_gate_scan.csv", "debate_lens_band_diagnosis.csv",
          "debate_lens_n150_summary.csv"]:
    print("  experiments/results/%s" % f)
print("=" * 72)
