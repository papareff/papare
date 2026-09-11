"""反讽任务辩论消融实验（独立运行，避免与推理者模型挤占显存）。

对比：
  - with_debate：完整管线（正反方辩论者 + 微调裁决者投票制裁决）
  - no_debate：仅微调裁决者独立判决（消融基线）
样本：从测试集中分层抽样（与 FS 实验相同的抽样逻辑但独立批次）。
标签约定：pro = ironic，con = not ironic。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import random
import numpy as np
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

DEBATE_SAMPLES = 150
SEED = 42

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config['task'] = 'irony'
config['debate']['rounds'] = 2

test_df = pd.read_csv("data/raw/irony_test.csv")
test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce')
test_df = test_df.dropna(subset=['label', 'text'])
test_df['label'] = test_df['label'].astype(int)

deb_strat = (
    test_df.groupby('label', group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), DEBATE_SAMPLES // 2), random_state=SEED))
)
deb_samples = deb_strat.reset_index(drop=True).to_dict('records')
random.seed(SEED)
random.shuffle(deb_samples)
print(f"辩论阶段样本: {len(deb_samples)}")


def is_correct(true_label, pred):
    return (true_label == 1 and pred == "pro") or (true_label == 0 and pred == "con")


os.makedirs("experiments/results", exist_ok=True)

# 构建带辩论管线（裁决者=微调模型）
debate_config = dict(config)
debate_config['judge'] = {"name": "./models/irony_distilbert", "device": "cuda", "max_length": 128}
pipeline = InferencePipeline(debate_config, use_debate=True, retriever=None, num_shots=0)

summary = []

print(f"\n===== with_debate（{len(deb_samples)} 条）=====")
rows, correct = [], 0
for i, s in enumerate(deb_samples):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    r = pipeline.run(s)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - t0
    ok = is_correct(s['label'], r["decision"])
    correct += int(ok)
    judge_vote = next((v["vote"] for v in r.get("vote_detail", []) if v["voter"] == "judge"), None)
    rows.append({"sample_id": i, "true_label": s['label'], "predicted": r["decision"],
                 "judge_vote": judge_vote,
                 "judge_confidence": r.get("judge_confidence"),
                 "pro_vote": r.get("debate_votes", {}).get("pro"),
                 "con_vote": r.get("debate_votes", {}).get("con"),
                 "latency_sec": latency, "correct": ok})
    print(f"  [{i}] true={s['label']} pred={r['decision']} judge={judge_vote} votes="
          f"{r.get('debate_votes', {}).get('pro')}/{r.get('debate_votes', {}).get('con')} ({latency:.1f}s)")
acc_d = correct / len(deb_samples)
pd.DataFrame(rows).to_csv("experiments/results/irony_debate_with.csv", index=False)
print(f"  → with_debate: acc={acc_d:.2%}")
summary.append({"stage": "debate", "condition": "with_debate", "accuracy": acc_d,
                "n": len(deb_samples), "avg_latency": np.mean([r["latency_sec"] for r in rows])})

print(f"\n===== no_debate（{len(deb_samples)} 条）=====")
rows, correct = [], 0
for i, s in enumerate(deb_samples):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    r = pipeline.judge.make_decision(s, debate_votes=None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - t0
    ok = is_correct(s['label'], r["decision"])
    correct += int(ok)
    rows.append({"sample_id": i, "true_label": s['label'], "predicted": r["decision"],
                 "judge_confidence": r.get("judge_confidence"),
                 "latency_sec": latency, "correct": ok})
acc_nd = correct / len(deb_samples)
pd.DataFrame(rows).to_csv("experiments/results/irony_debate_without.csv", index=False)
print(f"  → no_debate: acc={acc_nd:.2%}")
summary.append({"stage": "debate", "condition": "no_debate", "accuracy": acc_nd,
                "n": len(deb_samples), "avg_latency": np.mean([r["latency_sec"] for r in rows])})

print("\n" + "=" * 60)
print("辩论消融结果")
print("=" * 60)
for item in summary:
    print(f"  {item['condition']:<14} acc={item['accuracy']:.2%} (n={item['n']}) "
          f"latency={item['avg_latency']:.2f}s")
pd.DataFrame(summary).to_csv("experiments/results/irony_debate_summary.csv", index=False)
print("结果已保存到 experiments/results/irony_debate_*.csv")
