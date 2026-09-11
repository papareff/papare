import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

def run_experiment(use_debate: bool, output_suffix: str):
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 传入 use_debate 参数
    pipeline = InferencePipeline(config, use_debate=use_debate)

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
        result = pipeline.run(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - start

        results.append({
            "sample_id": i,
            "text": sample["text"][:100] + "...",
            "true_label": sample["label"],
            "predicted": result["decision"],
            "strategy": result["strategy"],
            "latency_sec": latency,
            "debate_used": use_debate
        })
        print(f"True: {sample['label']}, Pred: {result['decision']}, Latency: {latency:.3f}s")

    os.makedirs("experiments/results", exist_ok=True)
    df = pd.DataFrame(results)
    filename = f"experiments/results/ablation_{'with_debate' if use_debate else 'no_debate'}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    print("Running ablation: WITH debate")
    run_experiment(use_debate=True, output_suffix="with_debate")
    print("\n" + "="*50 + "\n")
    print("Running ablation: WITHOUT debate")
    run_experiment(use_debate=False, output_suffix="no_debate")