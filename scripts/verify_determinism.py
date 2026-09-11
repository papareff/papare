"""确定性验证：贪心解码下，同样样本连跑两遍结果必须完全一致。

目的：修复前用 temperature=0.7 采样，同一样本每次运行判决可能不同，
导致 n=50 上的准确率差异（54% vs 72%）无法区分是方法改进还是随机波动。
改贪心解码（temperature=0）后，结果应完全可复现。

只用 8 条样本、连跑两遍，几分钟内给出结论。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

NUM = 8
SEED = 42
TEST_CSV = "data/raw/irony_test.csv"

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["task"] = "irony"
config["debate"]["rounds"] = 1
config["debate"]["neutral"] = True

test_df = pd.read_csv(TEST_CSV)
test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce")
test_df = test_df.dropna(subset=["label", "text"])
test_df["label"] = test_df["label"].astype(int)
full_batch = (
    test_df.groupby("label", group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), 150 // 2), random_state=SEED))
)
recs = full_batch.reset_index(drop=True).to_dict("records")
random.seed(SEED)
random.shuffle(recs)
samples = recs[:NUM]

debate_config = dict(config)
debate_config["judge"] = {"name": "./models/irony_distilbert", "device": "cuda", "max_length": 128}
pipeline = InferencePipeline(debate_config, use_debate=True, retriever=None, num_shots=0)

# 确认辩论者确实是贪心解码
print("辩论者 temperature =", pipeline.pro_debater.temperature,
      "（应为 0.0 才是贪心解码）\n", flush=True)


def run_pass(tag):
    out = []
    for i, s in enumerate(samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        r = pipeline.run(s)
        v = r.get("debate_votes", {})
        out.append((r["decision"], v.get("pro"), v.get("con")))
        print(f"  [{tag} #{i}] true={s['label']} pred={r['decision']} "
              f"votes={v.get('pro')}/{v.get('con')}", flush=True)
    return out


print("=== 第一遍 ===", flush=True)
pass1 = run_pass("A")
print("\n=== 第二遍（完全相同的输入与模型状态）===", flush=True)
pass2 = run_pass("B")

print("\n" + "=" * 60)
print("确定性验证结果")
print("=" * 60)
diff = [i for i, (a, b) in enumerate(zip(pass1, pass2)) if a != b]
if not diff:
    print(f"✅ 两遍结果完全一致（{NUM}/{NUM} 条判决与投票逐一相同）")
    print("   → 贪心解码生效，实验结果可复现，可进行 n=150 正式复核")
else:
    print(f"❌ 有 {len(diff)} 条不一致: {diff}")
    for i in diff:
        print(f"   #{i}: 第一遍 {pass1[i]} vs 第二遍 {pass2[i]}")
    print("   → 仍存在随机性，需检查解码参数")
