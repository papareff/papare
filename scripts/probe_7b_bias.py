"""7B 辩论者偏置探针（快速、低成本）。

核心假设检验：1.5B 辩论者在反讽任务上存在严重"反讽偏置"（几乎全部投 pro）。
换 7B 后偏置是否消失？

只测双方独立判断（各 1 次生成），不做辩护辩论，token 预算压到 40，
8 条样本约 10 分钟即可给出方向性结论，用以决定是否值得跑完整版（3~10 小时）。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import pandas as pd
import torch
from src.agents.debater_agent import DebaterAgent

SEED = 42
NUM_SAMPLES = 8
TEST_CSV = "data/raw/irony_test.csv"

# 与正式实验完全一致的配对批次
test_df = pd.read_csv(TEST_CSV)
test_df = test_df[pd.to_numeric(test_df['label'], errors='coerce').notna()].copy()
test_df['label'] = test_df['label'].astype(int)
full_batch = (
    test_df.groupby('label', group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), 150 // 2), random_state=SEED))
)
full_batch = full_batch.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
samples = full_batch.head(NUM_SAMPLES).reset_index(drop=True)

cfg = {
    "name": "Qwen/Qwen2-7B-Instruct",
    "device": "cuda",
    "max_length": 512,
    "temperature": 0.0,          # 探针用贪心解码，结果可复现
    "task": "irony",
    "max_memory": {0: "3GiB", "cpu": "25GiB"},
}

print(f"探针样本: {len(samples)} 条（正类 {int(samples['label'].sum())} / 负类 {int((samples['label']==0).sum())}）", flush=True)
print("加载 Qwen2-7B-Instruct...", flush=True)
t0 = time.time()
debater = DebaterAgent(cfg, stance="pro")
# 同一份模型跑两次独立判断（立场不影响 classify，classify 是中立的）
print(f"加载完成 {time.time()-t0:.1f}s\n", flush=True)

pro_votes, correct = 0, 0
lat = []
for i, row in samples.iterrows():
    sample = {"text": row["text"], "label": int(row["label"])}
    gold = "pro" if int(row["label"]) == 1 else "con"
    t1 = time.time()
    vote = debater.classify(sample, examples=None)
    dt = time.time() - t1
    lat.append(dt)
    ok = (vote == gold)
    pro_votes += int(vote == "pro")
    correct += int(ok)
    print(f"  [{i}] true={row['label']} vote={vote} {'✓' if ok else '✗'} ({dt:.1f}s) | {str(row['text'])[:55]}", flush=True)

print("\n" + "=" * 60)
print("7B 辩论者偏置探针结果")
print("=" * 60)
print(f"  独立判断准确率: {correct}/{len(samples)} = {correct/len(samples)*100:.1f}%")
print(f"  投 pro（反讽）比例: {pro_votes}/{len(samples)} = {pro_votes/len(samples)*100:.1f}%")
print(f"  平均单条耗时: {sum(lat)/len(lat):.1f}s")
print(f"\n对比基准：1.5B 零样本准确率 50.33%，投 pro 比例 99.7%（299/300，严重偏置）")
print("判读：若 7B 投 pro 比例显著下降且准确率提升 → 偏置可被模型规模消除，值得跑完整版")
print("      若仍接近全投 pro → 换更大模型也无效，应转向门控融合方案")
