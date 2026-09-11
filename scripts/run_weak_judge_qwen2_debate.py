import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

# 加载配置
with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 修改配置：辩论模型用Qwen2，裁决器用GPT-2
debater_config = {
    "name": "Qwen/Qwen2-1.5B-Instruct",
    "device": "cuda",
    "max_length": 512,
    "temperature": 0.7
}
judge_config = {
    "name": "gpt2",
    "device": "cuda",
    "max_length": 128,
    "temperature": 0.1
}

# 手动注入配置到pipeline（为了不改动主代码，我们直接修改config字典，并在pipeline中覆盖）
# 但更方便：直接修改pipeline初始化时传入的config
config['model'] = debater_config   # 辩论模型
# 但judge_config需要单独处理，我们修改pipeline的__init__来接受两个配置？或者直接hack？
# 最简单的做法：复制pipeline的__init__代码，硬编码debater和judge的配置。
# 但为了快速，我建议直接修改pipeline的__init__，添加判断：如果use_debate=True且config['model']['name']包含'Qwen'，则使用特定配置。
# 更干净：我们新建一个函数，在外部创建DebaterAgent和JudgeAgent，再传入pipeline。
# 但为了省事，我们就直接修改pipeline的__init__，使其从config中读取两个独立的key：debater_model和judge_model。

# 但最快的方式：我们直接修改pipeline的代码，加两个参数，但是不破坏原有逻辑。
# 我决定：在run_experiment中，直接实例化DebaterAgent和JudgeAgent，然后传给pipeline，但需要修改pipeline支持外部传入。
# 算了，我们还是按照“方向一”的精神，直接在现有pipeline上改配置，让pipeline初始化时使用debater_config和judge_config。
# 简单改法：在pipeline的__init__中，从config读取debater和judge分别配置。
# 我先给你提供修改pipeline的方法，然后你再运行。

# 由于你需要快速跑，我建议直接修改pipeline.py的__init__，添加如下逻辑：
# self.pro_debater = DebaterAgent(config.get("debater_model", config.get("model", {})), stance="pro")
# self.judge = JudgeAgent(config.get("judge_model", config.get("model", {})))
# 这样config中可包含debater_model和judge_model两个key。
# 但为了不破坏现有实验，我们可以让config中同时存在这两个key。