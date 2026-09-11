"""按裁决者置信度分段，诊断辩论增益的方向（离线，零 GPU）。

输出 experiments/results/gate_subset_diagnosis.csv，供文档引用可追溯。
"""
import os
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "experiments", "results")
RES = os.path.normpath(RES)

PAIRS = [
    ("反讽·预设立场", "irony_debate_with.csv", "irony_debate_without.csv"),
    ("反讽·立场中立化", "irony_debate_neutral_with.csv", "irony_debate_neutral_without.csv"),
]

BINS = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
LABELS = ["[0,0.5)", "[0.5,0.55)", "[0.55,0.6)", "[0.6,0.65)", "[0.65,0.7)",
          "[0.7,0.75)", "[0.75,0.8)", "[0.8,0.85)", "[0.85,0.9)", "[0.9,0.95)", "[0.95,1.0]"]

rows = []
for name, wf, wof in PAIRS:
    with_df = pd.read_csv(os.path.join(RES, wf))
    wo_df = pd.read_csv(os.path.join(RES, wof))
    merged = with_df.merge(
        wo_df[["sample_id", "predicted", "correct"]].rename(
            columns={"predicted": "pred_wo", "correct": "correct_wo"}),
        on="sample_id", how="inner",
    )
    merged["bin"] = pd.cut(merged["judge_confidence"], bins=BINS, labels=LABELS, right=False)
    for b, grp in merged.groupby("bin", observed=False):
        if len(grp) == 0:
            continue
        acc_w = pd.to_numeric(grp["correct"], errors="coerce").mean() * 100
        acc_wo = pd.to_numeric(grp["correct_wo"], errors="coerce").mean() * 100
        rows.append({
            "dataset": name, "confidence_bin": str(b), "n": len(grp),
            "judge_acc_pct": round(acc_wo, 2), "debate_acc_pct": round(acc_w, 2),
            "delta_pp": round(acc_w - acc_wo, 2),
        })
    print(f"{name}: 合并样本 {len(merged)} 条")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "gate_subset_diagnosis.csv"), index=False)
print("\n", out.to_string(index=False))
print("\n已保存 gate_subset_diagnosis.csv")
