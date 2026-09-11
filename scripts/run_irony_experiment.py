"""反讽任务完整实验矩阵（SemEval-2018 Task 3A / tweet_eval irony）。

阶段 A：少样本检索实验（模块 II）
  - 推理者（Qwen2-1.5B）在 ZS / 1-shot / 3-shot / 5-shot 下做中立分类
  - 少样本示例来自 TF-IDF 检索器（示例池=训练集）
  - 对照：微调裁决者基线（irony_distilbert）
阶段 B：辩论消融实验（模块 III）
  - 完整管线（辩论 + 投票制裁决）vs 仅裁决者
  - 验证辩论机制在难任务上是否能纠正裁决者与推理者的错误

标签约定：pro = ironic（反讽），con = not ironic（非反讽）。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import pandas as pd
import torch
import yaml
from src.agents.debater_agent import DebaterAgent
from src.agents.judge_agent import JudgeAgent
from src.pipeline.inference_pipeline import InferencePipeline
from src.retriever import FewShotRetriever

FS_SAMPLES = 300       # FS 每条件样本数
DEBATE_SAMPLES = 150   # 辩论每条件样本数（辩论较慢）
SEED = 42
SHOTS = [0, 1, 3, 5]

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config['task'] = 'irony'
config['debate']['rounds'] = 2

# ---------- 数据 ----------
test_df = pd.read_csv("data/raw/irony_test.csv")
test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce')
test_df = test_df.dropna(subset=['label', 'text'])
test_df['label'] = test_df['label'].astype(int)

# FS 阶段：分层抽样
fs_strat = (
    test_df.groupby('label', group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), FS_SAMPLES // 2), random_state=SEED))
)
fs_samples = fs_strat.reset_index(drop=True).to_dict('records')

# 辩论阶段：独立分层抽样（与 FS 不同样本，避免同一批样本反复用）
test_rest = test_df[~test_df.index.isin(fs_strat.index)]
deb_strat = (
    test_rest.groupby('label', group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), DEBATE_SAMPLES // 2), random_state=SEED))
)
deb_samples = deb_strat.reset_index(drop=True).to_dict('records')

# 打乱顺序，避免同类样本连续出现误导中途进度
import random
random.seed(SEED)
random.shuffle(fs_samples)
random.shuffle(deb_samples)

print(f"FS 阶段样本: {len(fs_samples)}，辩论阶段样本: {len(deb_samples)}")

# ---------- 模块加载 ----------
print("加载推理者（Qwen2-1.5B，irony 任务）...")
reasoner = DebaterAgent({
    "name": "Qwen/Qwen2-1.5B-Instruct",
    "device": "cuda",
    "max_length": 512,
    "temperature": 0.7,
    "task": "irony",
}, stance="pro")

print("加载检索器（示例池=训练集）...")
retriever = FewShotRetriever("data/raw/irony_train.csv", k=max(SHOTS))

print("加载微调裁决者基线（irony_distilbert）...")
judge_baseline = JudgeAgent({
    "name": "./models/irony_distilbert",
    "device": "cuda",
    "max_length": 128,
})


def is_correct(true_label, pred):
    return (true_label == 1 and pred == "pro") or (true_label == 0 and pred == "con")


os.makedirs("experiments/results", exist_ok=True)
summary = []

# ================= 阶段 A：少样本 =================
for k in SHOTS:
    cond = "zero_shot" if k == 0 else f"{k}_shot"
    print(f"\n===== [FS] {cond}（{len(fs_samples)} 条）=====")
    rows, correct = [], 0
    for i, s in enumerate(fs_samples):
        examples = retriever.retrieve(s['text'], k=k) if k > 0 else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        pred = reasoner.classify(s, examples=examples)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0
        if pred is None:  # 解析失败保底：用裁决者
            pred = judge_baseline.make_decision(s, debate_votes=None)["decision"]
        ok = is_correct(s['label'], pred)
        correct += int(ok)
        rows.append({"sample_id": i, "true_label": s['label'], "predicted": pred,
                     "strategy": cond, "latency_sec": latency, "correct": ok})
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(fs_samples)}] 当前准确率 {correct/(i+1):.2%}")
    acc = correct / len(fs_samples)
    pd.DataFrame(rows).to_csv(f"experiments/results/irony_fs_{cond}.csv", index=False)
    preds = [r["predicted"] for r in rows]
    print(f"  → {cond}: acc={acc:.2%}, pro={preds.count('pro')}, con={preds.count('con')}")
    summary.append({"stage": "fs", "condition": cond, "accuracy": acc,
                    "n": len(fs_samples),
                    "avg_latency": np.mean([r["latency_sec"] for r in rows])})

# FS 微调基线（快，全量测试集）
print(f"\n===== [FS] finetuned_baseline（全量 {len(test_df)} 条）=====")
rows, correct = [], 0
for i, s in enumerate(test_df.to_dict('records')):
    pred = judge_baseline.make_decision(s, debate_votes=None)["decision"]
    ok = is_correct(s['label'], pred)
    correct += int(ok)
    rows.append({"sample_id": i, "true_label": s['label'], "predicted": pred,
                 "strategy": "finetuned_baseline", "correct": ok})
acc_ft = correct / len(test_df)
pd.DataFrame(rows).to_csv("experiments/results/irony_fs_finetuned_baseline.csv", index=False)
print(f"  → finetuned_baseline: acc={acc_ft:.2%}（全量测试集）")
summary.append({"stage": "fs", "condition": "finetuned_baseline", "accuracy": acc_ft,
                "n": len(test_df), "avg_latency": None})

# ================= 阶段 B：辩论消融 =================
# 构建带辩论管线（裁决者用微调模型）
print("\n加载辩论管线（裁决者=irony_distilbert）...")
debate_config = dict(config)
debate_config['judge'] = {"name": "./models/irony_distilbert", "device": "cuda", "max_length": 128}
pipeline = InferencePipeline(debate_config, use_debate=True, retriever=None, num_shots=0)

print(f"\n===== [辩论] with_debate（{len(deb_samples)} 条）=====")
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
    print(f"  [{i}] true={s['label']} pred={r['decision']} votes="
          f"{r.get('debate_votes', {}).get('pro')}/{r.get('debate_votes', {}).get('con')} ({latency:.1f}s)")
acc_d = correct / len(deb_samples)
pd.DataFrame(rows).to_csv("experiments/results/irony_debate_with.csv", index=False)
print(f"  → with_debate: acc={acc_d:.2%}")
summary.append({"stage": "debate", "condition": "with_debate", "accuracy": acc_d,
                "n": len(deb_samples), "avg_latency": np.mean([r["latency_sec"] for r in rows])})

print(f"\n===== [辩论] no_debate（{len(deb_samples)} 条）=====")
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

# ---------- 汇总 ----------
print("\n" + "=" * 60)
print("反讽任务实验汇总")
print("=" * 60)
for item in summary:
    lat = f"{item['avg_latency']:.2f}s" if item['avg_latency'] else "-"
    print(f"  [{item['stage']:6}] {item['condition']:<22} acc={item['accuracy']:.2%} "
          f"(n={item['n']}) latency={lat}")
pd.DataFrame(summary).to_csv("experiments/results/irony_summary.csv", index=False)
print("\n结果已保存到 experiments/results/irony_*.csv")
