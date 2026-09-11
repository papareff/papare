import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import yaml
from src.pipeline.inference_pipeline import InferencePipeline
from src.agents.judge_agent import JudgeAgent

def run_with_debate(num_samples=100):
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 使用本地微调好的模型作为裁决者
    judge_config = {
        "name": "./models/agnews_distilbert",
        "device": "cuda",
        "max_length": 256
    }
    pipeline = InferencePipeline(config, use_debate=True)
    pipeline.judge = JudgeAgent(judge_config)
    test_df = pd.read_csv("data/raw/agnews_test.csv").head(num_samples)
    label_map = {"world": 0, "sports": 1, "business": 2, "tech": 3}
    correct = 0
    for i, row in test_df.iterrows():
        result = pipeline.run(row.to_dict())
        pred = label_map.get(result["decision"], -1)
        if pred == row["label"]:
            correct += 1
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_samples}")
    print(f"✅ AG News with Debate Accuracy: {correct/num_samples:.4f} ({correct}/{num_samples})")

if __name__ == "__main__":
    run_with_debate(100)