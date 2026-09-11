"""少样本（FS）实验：ZS / 1-shot / 3-shot / 5-shot 推理者 + 微调基线对比。

设计说明：
- 推理者（Qwen2-1.5B）做中立分类（debater.classify），少样本示例来自
  模块 II 检索器（TF-IDF，示例池仅用训练集，无数据泄漏）。
- 微调基线：微调好的 DistilBERT 裁决者（独立分类）。
- 采样：从测试集分层抽样 300 条（正负类均衡），固定随机种子保证可复现。
- 一次加载模型，依次跑所有条件，避免重复加载。
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
from src.retriever import FewShotRetriever

NUM_SAMPLES = 300
SEED = 42
SHOTS = [0, 1, 3, 5]  # 0 = zero-shot

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ---------- 数据：分层抽样 ----------
test_df = pd.read_csv("data/splits/sst2_test.csv")
test_df = test_df.rename(columns={"sentence": "text"})
test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce')
test_df = test_df.dropna(subset=['label', 'text'])
test_df['label'] = test_df['label'].astype(int)
# 分层抽样：每个类别取 NUM_SAMPLES//2 条
strat = (
    test_df.groupby('label', group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), NUM_SAMPLES // 2), random_state=SEED))
)
samples = strat.reset_index(drop=True).to_dict('records')
print(f"抽样样本数: {len(samples)}（正类 {sum(s['label'] for s in samples)} / "
      f"负类 {len(samples) - sum(s['label'] for s in samples)}）")

# ---------- 模块加载（一次性） ----------
print("加载推理者（Qwen2-1.5B）...")
reasoner_config = {
    "name": "Qwen/Qwen2-1.5B-Instruct",
    "device": "cuda",
    "max_length": 512,
    "temperature": 0.7,
    "task": "sentiment",
}
reasoner = DebaterAgent(reasoner_config, stance="pro")  # classify() 不依赖立场

print("加载少样本检索器（模块 II，TF-IDF，示例池=训练集）...")
retriever = FewShotRetriever("data/splits/sst2_train.csv", k=max(SHOTS))

print("加载微调基线（DistilBERT-SST2）...")
judge = JudgeAgent({
    "name": "distilbert-base-uncased-finetuned-sst-2-english",
    "device": "cuda",
    "max_length": 512,
})


def is_correct(true_label, pred):
    return (true_label == 1 and pred == "pro") or (true_label == 0 and pred == "con")


# ---------- 实验主体 ----------
os.makedirs("experiments/results", exist_ok=True)
summary = []

for k in SHOTS:
    cond_name = "zero_shot" if k == 0 else f"{k}_shot"
    print(f"\n===== 条件: {cond_name}（{len(samples)} 条）=====")
    rows = []
    correct = 0
    for i, s in enumerate(samples):
        examples = retriever.retrieve(s['text'], k=k) if k > 0 else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        pred = reasoner.classify(s, examples=examples)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0
        # 解析失败回退为裁决者判断（保底）
        if pred is None:
            pred = judge.make_decision(s, debate_votes=None)["decision"]
        ok = is_correct(s['label'], pred)
        correct += int(ok)
        rows.append({
            "sample_id": i, "true_label": s['label'], "predicted": pred,
            "strategy": cond_name, "latency_sec": latency, "correct": ok,
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(samples)}] 当前准确率 {correct/(i+1):.2%}")
    acc = correct / len(samples)
    pd.DataFrame(rows).to_csv(f"experiments/results/fs_{cond_name}.csv", index=False)
    preds = [r["predicted"] for r in rows]
    lats = np.mean([r["latency_sec"] for r in rows])
    print(f"  → {cond_name}: acc={acc:.2%}, pro={preds.count('pro')}, con={preds.count('con')}, 平均延迟={lats:.2f}s")
    summary.append({"condition": cond_name, "accuracy": acc, "avg_latency": lats})

# ---------- 微调基线 ----------
print(f"\n===== 条件: finetuned_baseline（{len(samples)} 条）=====")
rows = []
correct = 0
for i, s in enumerate(samples):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    pred = judge.make_decision(s, debate_votes=None)["decision"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - t0
    ok = is_correct(s['label'], pred)
    correct += int(ok)
    rows.append({
        "sample_id": i, "true_label": s['label'], "predicted": pred,
        "strategy": "finetuned_baseline", "latency_sec": latency, "correct": ok,
    })
acc_ft = correct / len(samples)
pd.DataFrame(rows).to_csv("experiments/results/fs_finetuned_baseline.csv", index=False)
print(f"  → finetuned_baseline: acc={acc_ft:.2%}")
summary.append({"condition": "finetuned_baseline", "accuracy": acc_ft,
                "avg_latency": np.mean([r["latency_sec"] for r in rows])})

# ---------- 汇总 ----------
print("\n" + "=" * 60)
print("少样本实验汇总")
print("=" * 60)
for item in summary:
    print(f"  {item['condition']:<22} acc={item['accuracy']:.2%}  latency={item['avg_latency']:.2f}s")
pd.DataFrame(summary).to_csv("experiments/results/fs_summary.csv", index=False)
print("\n结果已保存到 experiments/results/fs_*.csv")
