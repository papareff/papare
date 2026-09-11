import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

def run_test(num_samples=100):
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = InferencePipeline(config, use_debate=True)

    test_df = pd.read_csv("data/splits/sst2_test.csv")
    test_df = test_df.rename(columns={"sentence": "text"})
    test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce').fillna(0).astype(int)
    samples = test_df.head(num_samples).to_dict('records')

    correct = 0
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        start = time.time()
        result = pipeline.run(sample)
        latency = time.time() - start
        pred = result["decision"]
        true = sample["label"]
        correct += (true == 1 and pred == "pro") or (true == 0 and pred == "con")
        print(f"True: {true}, Pred: {pred}, Latency: {latency:.2f}s")

    accuracy = correct / num_samples
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{num_samples})")

if __name__ == "__main__":
    run_test(100)