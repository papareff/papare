"""自然类别比例测试集：视角分化辩论完整流程复核。

目的：验证事后分层估算的结论（均衡集 +18.67pp → 自然分布 +13.43pp，增益衰减 72% 保留）。
抽样：从完整测试集（反讽 311 / 非反讽 473）按自然分布随机抽样 n=200，不做分层均衡。
对照：同一批样本上的纯裁决者基线。

贪心解码，可复现。预计 200 × 23s ≈ 77 分钟。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import time
import numpy as np
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

NUM = 200
SEED = 42
ROUNDS = 1
TEST_CSV = "data/raw/irony_test.csv"
OUT_DIR = "experiments/results"
LABEL_VOTE = {1: "pro", 0: "con"}

os.makedirs(OUT_DIR, exist_ok=True)

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["task"] = "irony"
config["debate"]["rounds"] = ROUNDS
config["debate"]["neutral"] = True

# ── 自然分布随机抽样（不分层，不均衡）──
test_df = pd.read_csv(TEST_CSV)
test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce")
test_df = test_df.dropna(subset=["label", "text"])
test_df["label"] = test_df["label"].astype(int)

samples = (test_df.sample(n=NUM, random_state=SEED)
           .reset_index(drop=True).to_dict("records"))
n_pro = sum(1 for s in samples if int(s["label"]) == 1)
print(f"样本: {len(samples)} 条（反讽 {n_pro} / 非反讽 {len(samples)-n_pro}，"
      f"反讽占比 {n_pro/len(samples)*100:.1f}%）", flush=True)
print(f"完整测试集自然分布: 反讽 {(test_df.label==1).mean()*100:.1f}%", flush=True)

debate_config = dict(config)
debate_config["judge"] = {
    "name": "./models/irony_distilbert", "device": "cuda", "max_length": 128
}
pipeline = InferencePipeline(debate_config, use_debate=True, retriever=None, num_shots=0)


def correct_of(tl, pred):
    return pred == LABEL_VOTE[int(tl)]


# ══════════ 1) 视角分化辩论 ══════════
print(f"\n===== 视角分化辩论（n={len(samples)}）=====", flush=True)
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
        print(f"  [{i+1}/{len(samples)}] 当前准确率 "
              f"{np.mean([x['correct'] for x in rows])*100:.2f}%  "
              f"已用 {el/60:.1f}min  剩余 {eta/60:.1f}min", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/natural200_debate_with.csv", index=False, encoding="utf-8-sig")

# ══════════ 2) 纯裁决者基线 ══════════
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
base_df.to_csv(f"{OUT_DIR}/natural200_judge_only.csv", index=False, encoding="utf-8-sig")


def se(pct, n):
    return math.sqrt(pct * (100 - pct) / n)


acc_d = df.correct.mean() * 100
acc_j = base_df.correct.mean() * 100
delta = acc_d - acc_j

merged = df.merge(base_df[["sample_id", "correct"]].rename(columns={"correct": "jc"}),
                  on="sample_id")
disc = merged[merged.correct != merged.jc]
b_cnt = int((disc.correct == 1).sum())
c_cnt = int((disc.jc == 1).sum())
chi2 = (abs(b_cnt - c_cnt) - 1) ** 2 / (b_cnt + c_cnt) if (b_cnt + c_cnt) else 0.0
se_pair = (math.sqrt((b_cnt + c_cnt) - (b_cnt - c_cnt) ** 2 / (b_cnt + c_cnt))
           / len(merged) * 100) if (b_cnt + c_cnt) else 0.0

print("\n" + "=" * 78)
print("自然分布复核结果（n=%d，反讽占比 %.1f%%）" % (NUM, n_pro / NUM * 100))
print("=" * 78)
print("① 准确率对照")
print("   视角分化辩论 : %.2f%%  ±%.2fpp" % (acc_d, se(acc_d, NUM) * 1.96))
print("   纯裁决者     : %.2f%%  ±%.2fpp" % (acc_j, se(acc_j, NUM) * 1.96))
print("   辩论增益     : %+.2fpp  ±%.2fpp" % (delta, se_pair * 1.96))
print("   McNemar χ²=%.3f → %s" % (chi2, "显著" if chi2 > 3.841 else "不显著"))
print("")
print("② 分类别命中（对比均衡集）")
print("   %-12s %10s %10s %10s" % ("类别", "裁决者", "辩论", "差值"))
for lab, name in [(1, "真反讽"), (0, "真非反讽")]:
    s1 = merged[merged.true_label == lab]
    print("   %-12s %9.2f%% %9.2f%% %+9.2fpp" % (
        name, s1.jc.mean() * 100, s1.correct.mean() * 100,
        (s1.correct.mean() - s1.jc.mean()) * 100))
print("")
print("③ 与均衡集对比（核心验证）")
print("   %-22s %10s %10s %10s" % ("测试集", "裁决者", "辩论", "增益"))
print("   %-22s %9.2f%% %9.2f%% %+9.2fpp" % ("均衡 50:50 (n=150)", 59.33, 78.00, 18.67))
print("   %-22s %9.2f%% %9.2f%% %+9.2fpp" % (
    "自然分布 (n=%d)" % NUM, acc_j, acc_d, delta))
print("   事后分层估算预期        %9.2f%% %9.2f%% %+9.2fpp" % (61.40, 74.83, 13.43))
print("")
print("   估算 vs 实测: 裁决者 %+.2fpp, 辩论 %+.2fpp, 增益 %+.2fpp" % (
    acc_j - 61.40, acc_d - 74.83, delta - 13.43))
print("   → %s" % ("估算与实测一致，偏置抵消假设成立" if abs(delta - 13.43) < 6 else
                     "实测与估算有偏差，需进一步分析"))
print("")
print("④ 判决分布")
print("   真实反讽占比 %.1f%%   辩论判 pro %.1f%%   裁决者判 pro %.1f%%" % (
    n_pro / NUM * 100, (df.predicted == "pro").mean() * 100,
    (base_df.predicted == "pro").mean() * 100))
print("   双方投票一致率: %.1f%%" % (df.votes_agree.mean() * 100))
print("")
print("⑤ 门控阈值扫描（自然分布下）")
lat_d = float(np.mean(lats))
print("   %-8s %10s %10s %12s %12s" % ("τ", "准确率", "触发率", "加权延迟", "vs纯裁决者"))
best = None
gate_rows = []
for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]:
    low = df.judge_confidence < tau
    pred = np.where(low, df.predicted, df.judge_vote)
    a = (pd.Series(pred, index=df.index) == df.gold).mean() * 100
    trig = low.mean() * 100
    wl = trig / 100 * lat_d + (1 - trig / 100) * 0.01
    gate_rows.append({"tau": tau, "accuracy": a, "trigger_rate": trig, "weighted_latency": wl})
    print("   %-8.2f %9.2f%% %9.1f%% %11.2fs %+11.2fpp" % (tau, a, trig, wl, a - acc_j))
    if best is None or a > best["accuracy"]:
        best = {"tau": tau, "accuracy": a, "trigger_rate": trig, "weighted_latency": wl}
pd.DataFrame(gate_rows).to_csv(f"{OUT_DIR}/natural200_gate_scan.csv",
                               index=False, encoding="utf-8-sig")
print("   ★ 最优 τ=%.2f: %.2f%%（触发率 %.1f%%，延迟 %.2fs）" % (
    best["tau"], best["accuracy"], best["trigger_rate"], best["weighted_latency"]))
print("     vs 纯裁决者 %+.2fpp，vs 全程辩论 %+.2fpp，延迟节省 %.0f%%" % (
    best["accuracy"] - acc_j, best["accuracy"] - acc_d,
    (1 - best["weighted_latency"] / lat_d) * 100))

pd.DataFrame([{
    "n": NUM, "irony_ratio_pct": round(n_pro / NUM * 100, 2), "rounds": ROUNDS,
    "acc_debate_pct": round(acc_d, 2), "acc_judge_pct": round(acc_j, 2),
    "delta_pp": round(delta, 2), "delta_ci95_pp": round(se_pair * 1.96, 2),
    "mcnemar_chi2": round(chi2, 3), "significant": bool(chi2 > 3.841),
    "vote_agree_pct": round(df.votes_agree.mean() * 100, 2),
    "pred_pro_pct": round((df.predicted == "pro").mean() * 100, 2),
    "avg_latency_sec": round(lat_d, 2),
    "best_tau": best["tau"], "best_gate_acc_pct": round(best["accuracy"], 2),
    "best_gate_trigger_pct": round(best["trigger_rate"], 2),
    "balanced_delta_pp": 18.67, "poststrat_est_delta_pp": 13.43,
}]).to_csv(f"{OUT_DIR}/natural200_summary.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 78)
print("结果文件: natural200_debate_with.csv / natural200_judge_only.csv /")
print("          natural200_gate_scan.csv / natural200_summary.csv")
print("=" * 78)
