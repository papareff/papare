import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

# 加载配置，并强制使用 gpt2
with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config['model']['name'] = "gpt2"  # 弱裁决者
config['model']['max_length'] = 128

# 创建 pipeline，禁用辩论
pipeline = InferencePipeline(config, use_debate=False)

# 加载 SST-2 测试集
test_df = pd.read_csv("data/splits/sst2_test.csv")
test_df = test_df.rename(columns={"sentence": "text"})
test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce').fillna(0).astype(int)
samples = test_df.head(20).to_dict('records')

results = []
for i, sample in enumerate(samples):
    print(f"\n--- Sample {i+1}/{len(samples)} ---")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    result = pipeline.run(sample)   # 因为 use_debate=False，不会调用辩论
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - start
    results.append({
        "sample_id": i,
        "text": sample["text"][:100] + "...",
        "true_label": sample["label"],
        "predicted": result["decision"],
        "latency_sec": latency,
        "debate_used": False
    })
    print(f"True: {sample['label']}, Pred: {result['decision']}, Latency: {latency:.3f}s")

os.makedirs("experiments/results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("experiments/results/weak_judge_no_debate.csv", index=False)
print("Results saved to experiments/results/weak_judge_no_debate.csv")