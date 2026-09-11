"""冒烟测试：修复后的 classify 方法在 3 条样本上验证返回值与准确率。"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import yaml
from src.agents.debater_agent import DebaterAgent
from src.retriever import FewShotRetriever

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

reasoner = DebaterAgent({
    "name": "Qwen/Qwen2-1.5B-Instruct",
    "device": "cuda",
    "max_length": 512,
    "temperature": 0.7,
    "task": "sentiment",
}, stance="pro")

retriever = FewShotRetriever("data/splits/sst2_train.csv", k=5)

test_df = pd.read_csv("data/splits/sst2_test.csv").rename(columns={"sentence": "text"})
samples = test_df.head(6).to_dict('records')

for k in [0, 3]:
    print(f"\n--- {k}-shot ---")
    for i, s in enumerate(samples):
        examples = retriever.retrieve(s['text'], k=k) if k > 0 else None
        pred = reasoner.classify(s, examples=examples)
        true = int(s['label'])
        ok = (true == 1 and pred == "pro") or (true == 0 and pred == "con")
        print(f"[{i}] true={true} pred={pred} {'✓' if ok else '✗'}")
