import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.pipeline.inference_pipeline import InferencePipeline

# 加载配置
with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 创建流水线
pipeline = InferencePipeline(config)

# 测试样本（模拟）
test_sample = {"text": "The movie was absolutely fantastic. The acting was superb.", "label": 1}

# 运行
result = pipeline.run(test_sample)

print("Strategy:", result["strategy"])
print("Decision:", result["decision"])
print("Debate History:", result["debate_history"])