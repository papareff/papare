import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.agents.judge_agent import JudgeAgent

def run_baseline(num_samples=100):
    # 使用本地微调好的模型
    judge_config = {
        "name": "./models/agnews_distilbert",
        "device": "cuda",
        "max_length": 256
    }
    judge = JudgeAgent(judge_config)
    test_df = pd.read_csv("data/raw/agnews_test.csv").head(num_samples)
    label_map = {"world": 0, "sports": 1, "business": 2, "tech": 3}
    correct = 0
    for i, row in test_df.iterrows():
        result = judge.make_decision(row.to_dict(), debate_votes=None)
        pred = result["decision"]
        if label_map.get(pred, -1) == row["label"]:
            correct += 1
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_samples}")
    print(f"✅ AG News Baseline Accuracy: {correct/num_samples:.4f} ({correct}/{num_samples})")

if __name__ == "__main__":
    run_baseline(100)
