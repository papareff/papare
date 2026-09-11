"""小规模验证（50 条）：新投票制管线下，带辩论 vs 不带辩论的配对对比。
- 带辩论：完整管线（Qwen2-1.5B 正反方辩论 + DistilBERT 裁决者投票）
- 不带辩论：仅裁决者独立判决（消融基线）
两组使用同一批样本，输出准确率、预测分布、辩论翻转裁决的案例分析。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

NUM_SAMPLES = 50

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config['task'] = 'sentiment'
config['debate']['rounds'] = 2

pipeline = InferencePipeline(config, use_debate=True)
judge = pipeline.judge  # 复用，避免二次加载

test_df = pd.read_csv("data/splits/sst2_test.csv")
test_df = test_df.rename(columns={"sentence": "text"})
samples = test_df.head(NUM_SAMPLES).to_dict('records')


def is_correct(true_label, pred):
    return (true_label == 1 and pred == "pro") or (true_label == 0 and pred == "con")


# ---------- 条件1：带辩论 ----------
print(f"===== 带辩论（{NUM_SAMPLES} 条）=====")
debate_rows = []
for i, s in enumerate(samples):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    r = pipeline.run(s)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - t0
    debate_rows.append({
        "sample_id": i,
        "true_label": s["label"],
        "predicted": r["decision"],
        "judge_vote": r.get("judge_vote"),
        "judge_confidence": r.get("judge_confidence"),
        "pro_vote": r.get("debate_votes", {}).get("pro"),
        "con_vote": r.get("debate_votes", {}).get("con"),
        "latency_sec": latency,
    })
    print(f"[{i}] true={s['label']} pred={r['decision']} judge={r.get('judge_vote')} "
          f"votes=pro:{r.get('debate_votes', {}).get('pro')}/con:{r.get('debate_votes', {}).get('con')} "
          f"({latency:.1f}s)")

# ---------- 条件2：不带辩论（仅裁决者）----------
print(f"\n===== 不带辩论（{NUM_SAMPLES} 条）=====")
nodebate_rows = []
for i, s in enumerate(samples):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    r = judge.make_decision(s, debate_votes=None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - t0
    nodebate_rows.append({
        "sample_id": i,
        "true_label": s["label"],
        "predicted": r["decision"],
        "judge_confidence": r.get("judge_confidence"),
        "latency_sec": latency,
    })

# ---------- 统计 ----------
acc_d = sum(is_correct(r["true_label"], r["predicted"]) for r in debate_rows) / NUM_SAMPLES
acc_n = sum(is_correct(r["true_label"], r["predicted"]) for r in nodebate_rows) / NUM_SAMPLES

print("\n" + "=" * 60)
print("结果汇总")
print("=" * 60)
print(f"带辩论准确率:   {acc_d:.2%}")
print(f"不带辩论准确率: {acc_n:.2%}")
print(f"辩论增益:       {acc_d - acc_n:+.2%}")

d_preds = [r["predicted"] for r in debate_rows]
n_preds = [r["predicted"] for r in nodebate_rows]
print(f"带辩论预测分布:   pro={d_preds.count('pro')}, con={d_preds.count('con')}")
print(f"不带辩论预测分布: pro={n_preds.count('pro')}, con={n_preds.count('con')}")

lat_d = sum(r["latency_sec"] for r in debate_rows) / NUM_SAMPLES
lat_n = sum(r["latency_sec"] for r in nodebate_rows) / NUM_SAMPLES
print(f"平均延迟: 带辩论={lat_d:.2f}s, 不带辩论={lat_n:.2f}s")

# 辩论翻转裁决者的案例分析
flips = [(r, n) for r, n in zip(debate_rows, nodebate_rows) if r["predicted"] != n["predicted"]]
print(f"\n辩论翻转裁决的案例数: {len(flips)}")
for r, n in flips:
    fix = "纠正✓" if is_correct(r["true_label"], r["predicted"]) else "改错✗"
    print(f"  样本{r['sample_id']}: true={r['true_label']} 裁决者={n['predicted']} → 辩论后={r['predicted']} [{fix}] "
          f"(votes pro:{r['pro_vote']}/con:{r['con_vote']}, conf={r['judge_confidence']:.3f})")

os.makedirs("experiments/results", exist_ok=True)
pd.DataFrame(debate_rows).to_csv("experiments/results/pilot50_with_debate.csv", index=False)
pd.DataFrame(nodebate_rows).to_csv("experiments/results/pilot50_no_debate.csv", index=False)
print("\n结果已保存到 experiments/results/pilot50_*.csv")
