import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

# 显存监控（如果可用）
try:
    from pynvml import *

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    has_nvml = True
except:
    has_nvml = False
    print("NVML not available, skipping GPU memory monitoring.")


def get_gpu_memory():
    if has_nvml:
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 ** 2  # MB
    return None


def run_experiment():
    # 加载配置
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化流水线
    pipeline = InferencePipeline(config)

    # 加载测试集（取前20个样本）
    test_df = pd.read_csv("data/splits/sst2_test.csv")
    # 重命名列 'sentence' 为 'text'，使其与 pipeline 期望一致
    test_df = test_df.rename(columns={"sentence": "text"})
    samples = test_df.head(20).to_dict('records')
    # 确保 label 列是整数类型（如果是字符串则转换）
    test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce').fillna(0).astype(int)
    samples = test_df.head(20).to_dict('records')

    results = []
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i + 1}/{len(samples)} ---")
        # 预热（首次运行可能较慢）
        if i == 0:
            _ = pipeline.run(sample)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # 计时
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        result = pipeline.run(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - start

        gpu_mem = get_gpu_memory()

        results.append({
            "sample_id": i,
            "text": sample["text"][:100] + "...",  # 改为 text
            "true_label": sample["label"],
            "predicted": result["decision"],
            "strategy": result["strategy"],
            "latency_sec": latency,
            "gpu_memory_mb": gpu_mem
        })
        print(f"True: {sample['label']}, Pred: {result['decision']}, Latency: {latency:.3f}s")

    # 保存结果
    os.makedirs("experiments/results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("experiments/results/zero_shot_sst2.csv", index=False)
    print(f"\nResults saved to experiments/results/zero_shot_sst2.csv")


if __name__ == "__main__":
    run_experiment()