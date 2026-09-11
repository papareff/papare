import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

# 创建配置字典
config = {
    "router": {
        "mode": "length_based",
        "length_threshold": 100
    },
    "debater_model": {
        "name": "Qwen/Qwen2-1.5B-Instruct",
        "device": "cuda",
        "max_length": 512,
        "temperature": 0.7
    },
    "judge_model": {
        "name": "gpt2",
        "device": "cuda",
        "max_length": 128,
        "temperature": 0.1
    },
    "debate": {
        "rounds": 2,
        "use_retrieval": False
    }
}

pipeline = InferencePipeline(config, use_debate=True)
# ----- 强制替换裁决者为 GPT-2 Large -----
from src.agents.judge_agent import JudgeAgent
judge_config = {"name": "gpt2-large", "device": "cuda", "max_length": 256}
pipeline.judge = JudgeAgent(judge_config)
# ------------------------------------------

# 加载测试集（只取前200个）
test_df = pd.read_csv("data/splits/sst2_test.csv")
test_df = test_df.rename(columns={"sentence": "text"})
test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce').fillna(0).astype(int)
samples = test_df.head(200).to_dict('records')

results = []
correct = 0
total = len(samples)

for i, sample in enumerate(samples):
    if i % 10 == 0:
        print(f"Processing {i}/{total}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    result = pipeline.run(sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - start
    pred = result["decision"]
    true = sample["label"]
    if (true == 1 and pred == "pro") or (true == 0 and pred == "con"):
        correct += 1
    results.append({
        "sample_id": i,
        "true_label": true,
        "predicted": pred,
        "latency_sec": latency,
    })
    if (i+1) % 10 == 0:
        acc = correct / (i+1)
        print(f"  Current accuracy: {acc:.4f}")

accuracy = correct / total
print(f"\nTotal samples: {total}, Correct: {correct}, Accuracy: {accuracy:.4f}")

os.makedirs("experiments/results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("experiments/results/qwen2_debate_gpt2_judge_200.csv", index=False)
print("Results saved to experiments/results/qwen2_debate_gpt2_judge_200.csv")