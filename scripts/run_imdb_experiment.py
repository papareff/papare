import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline


def run_imdb_experiment(use_debate, num_samples=100):
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 覆盖 max_length 为 512
    config['model']['max_length'] = 512

    pipeline = InferencePipeline(config, use_debate=use_debate)

    # 加载 IMDb 测试集
    test_df = pd.read_csv("data/raw/imdb_test.csv")
    test_df = test_df.head(num_samples)
    # 确保 label 为整数
    test_df['label'] = test_df['label'].astype(int)
    samples = test_df.to_dict('records')

    correct = 0
    total = len(samples)
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"Processing {i}/{total}")
        result = pipeline.run(sample)
        pred = result["decision"]
        true = sample["label"]
        # label: 0=neg, 1=pos; pro=pos, con=neg
        correct += (true == 1 and pred == "pro") or (true == 0 and pred == "con")

    accuracy = correct / total
    print(f"\n{'With' if use_debate else 'Without'} Debate - Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    print("Running IMDb 100-sample experiment...")
    print("=" * 50)
    acc_no_debate = run_imdb_experiment(use_debate=False, num_samples=100)
    print("=" * 50)
    acc_with_debate = run_imdb_experiment(use_debate=True, num_samples=100)
    print("=" * 50)
    print(f"Final Results: No Debate = {acc_no_debate:.4f}, With Debate = {acc_with_debate:.4f}")