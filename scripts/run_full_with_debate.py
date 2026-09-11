import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config['model']['name'] = "gpt2"
config['model']['max_length'] = 128
config['debate']['rounds'] = 2

pipeline = InferencePipeline(config, use_debate=True)

test_df = pd.read_csv("data/splits/sst2_test.csv")
test_df = test_df.rename(columns={"sentence": "text"})
test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce').fillna(0).astype(int)
samples = test_df.to_dict('records')  # 全部样本

results = []
correct = 0
total = len(samples)

for i, sample in enumerate(samples):
    if i % 100 == 0:
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

accuracy = correct / total
print(f"\nTotal samples: {total}, Correct: {correct}, Accuracy: {accuracy:.4f}")

os.makedirs("experiments/results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("experiments/results/full_with_debate.csv", index=False)
print("Results saved to experiments/results/full_with_debate.csv")