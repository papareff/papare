"""难度冒烟验证：推理者（Qwen2-1.5B）在反讽测试集上零样本分类的准确率。
目标：确认任务难度显著高于 SST-2（ZS 应落在约 55%~85% 的困难区间），
从而给检索与辩论留出纠错空间。仅跑前 N 条，快速判断。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from src.agents.debater_agent import DebaterAgent

NUM_SAMPLES = 40

reasoner = DebaterAgent({
    "name": "Qwen/Qwen2-1.5B-Instruct",
    "device": "cuda",
    "max_length": 512,
    "temperature": 0.7,
    "task": "irony",
}, stance="pro")

test_df = pd.read_csv("data/raw/irony_test.csv")
samples = test_df.head(NUM_SAMPLES).to_dict('records')


def is_correct(true_label, pred):
    return (true_label == 1 and pred == "pro") or (true_label == 0 and pred == "con")


correct = 0
preds = []
for i, s in enumerate(samples):
    pred = reasoner.classify(s, examples=None)
    preds.append(pred)
    ok = is_correct(s['label'], pred)
    correct += int(ok)
    print(f"[{i}] true={s['label']} pred={pred} {'✓' if ok else '✗'}")

acc = correct / len(samples)
print("\n" + "=" * 60)
print(f"反讽任务零样本准确率（前 {NUM_SAMPLES} 条）: {acc:.2%}")
print(f"预测分布: pro={preds.count('pro')}, con={preds.count('con')}")
print("目标区间: 55%~85%（太高=任务太简单，太低=模型能力不足）")
