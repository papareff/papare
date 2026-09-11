"""7B 辩论者验证实验（反讽任务，立场中立化）。

目的：验证辩论者换成更大模型（Qwen2-7B-Instruct）后，能否消除反讽先验偏置、使辩论增益转正。

配对设计：使用与 1.5B 实验完全相同的分层抽样批次（SEED=42, 150 条），取其前 50 条作为子集。
同一子集上可直接与已有的 1.5B 中立化结果（irony_debate_neutral_with.csv）配对对比。

硬件说明：本机 4GB 显存 + 32GB 内存，7B 以 FP16 跨设备放置（显存 3GiB，其余卸载到内存），
单条生成约 58.6s，50 条约需 2~3 小时。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import pandas as pd
import yaml

from src.pipeline.inference_pipeline import InferencePipeline

SEED = 42
NUM_SAMPLES = 50          # 从与 1.5B 相同的 150 条批次中取前 N 条
TEST_CSV = "data/raw/irony_test.csv"
OUT_DIR = "experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)

# 标签约定：pro = ironic（1），con = not ironic（0）
LABEL_TO_VOTE = {1: "pro", 0: "con"}

# ---- 抽样：与 run_irony_debate_neutral.py 完全一致的批次，再取前 NUM_SAMPLES ----
test_df = pd.read_csv(TEST_CSV)
test_df = test_df[pd.to_numeric(test_df['label'], errors='coerce').notna()].copy()
test_df['label'] = test_df['label'].astype(int)

full_batch = (
    test_df.groupby('label', group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), 150 // 2), random_state=SEED))
)
full_batch = full_batch.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
samples = full_batch.head(NUM_SAMPLES).reset_index(drop=True)
print(f"配对子集样本数: {len(samples)}（正类 {int(samples['label'].sum())} / 负类 {int((samples['label']==0).sum())}）", flush=True)

with open("configs/base.yaml", encoding="utf-8") as f:
    base_config = yaml.safe_load(f)

config = dict(base_config)
config["task"] = "irony"
config["judge"] = {"name": "./models/irony_distilbert"}
config["debate"] = {
    "rounds": 1,                      # 中立化：1 轮辩护足够（双方独立判断分歧后各辩护一次）
    "neutral": True,                  # 立场中立化
    "debater_model": "Qwen/Qwen2-7B-Instruct",
    "max_memory": {0: "3GiB", "cpu": "25GiB"},   # 4GB 显存放不下，溢出层卸载到内存
}

print("加载管线（辩论者=Qwen2-7B-Instruct，裁决者=irony_distilbert）...", flush=True)
t0 = time.time()
pipeline = InferencePipeline(config, use_debate=True)
print(f"加载完成，耗时 {time.time()-t0:.1f}s\n", flush=True)

# ---- 带辩论 ----
print(f"===== with_debate_7b（{len(samples)} 条）=====", flush=True)
rows, correct, latencies = [], 0, []
for i, row in samples.iterrows():
    sample = {"text": row["text"], "label": int(row["label"])}
    t1 = time.time()
    result = pipeline.run(sample)
    dt = time.time() - t1
    latencies.append(dt)
    pred = result["decision"]
    gold = LABEL_TO_VOTE[int(row["label"])]
    ok = (pred == gold)
    correct += int(ok)
    votes = result.get("debate_votes", {})
    judge_vote = result.get("vote_detail", {}).get("judge")
    rows.append({
        "idx": i, "text": row["text"], "label": int(row["label"]), "gold": gold,
        "pred": pred, "judge_vote": judge_vote,
        "pro_vote": votes.get("pro"), "con_vote": votes.get("con"),
        "judge_confidence": result.get("judge_confidence"),
        "correct": int(ok), "latency_s": round(dt, 2),
    })
    flag = "✓" if ok else "✗"
    print(f"  [{i}] true={row['label']} pred={pred} judge={judge_vote} "
          f"votes={votes.get('pro')}/{votes.get('con')} {flag} ({dt:.1f}s)", flush=True)

acc_debate = correct / len(samples) * 100
avg_lat = sum(latencies) / len(latencies)
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/irony_debate_7b_with.csv", index=False, encoding="utf-8-sig")

# ---- 不带辩论（同一子集，仅裁决者；速度快）----
print(f"\n===== no_debate（{len(samples)} 条，仅裁决者）=====", flush=True)
rows_nd, correct_nd = [], 0
for i, row in samples.iterrows():
    sample = {"text": row["text"], "label": int(row["label"])}
    res = pipeline.judge.make_decision(sample, debate_votes=None)
    pred = res["decision"]
    gold = LABEL_TO_VOTE[int(row["label"])]
    ok = (pred == gold)
    correct_nd += int(ok)
    rows_nd.append({"idx": i, "pred": pred, "gold": gold, "correct": int(ok),
                    "judge_confidence": res["judge_confidence"]})
acc_nodebate = correct_nd / len(samples) * 100
pd.DataFrame(rows_nd).to_csv(f"{OUT_DIR}/irony_debate_7b_nodebate.csv", index=False, encoding="utf-8-sig")

# ---- 投票偏置诊断（核心假设检验）----
df_with = pd.DataFrame(rows)
pro_votes = int((df_with["pro_vote"] == "pro").sum())
con_votes = int((df_with["con_vote"] == "pro").sum())

summary = pd.DataFrame([{
    "condition": "with_debate_7b_neutral",
    "accuracy": round(acc_debate, 2), "n": len(samples),
    "avg_latency_s": round(avg_lat, 2),
}, {
    "condition": "no_debate",
    "accuracy": round(acc_nodebate, 2), "n": len(samples),
    "avg_latency_s": 0.01,
}])
summary.to_csv(f"{OUT_DIR}/irony_debate_7b_summary.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("7B 辩论者验证结果（立场中立化，配对子集）")
print("=" * 60)
for _, r in summary.iterrows():
    print(f"  {r['condition']:<28} acc={r['accuracy']:.2f}% (n={r['n']}) latency={r['avg_latency_s']}s")
print(f"\n辩论增益: {acc_debate - acc_nodebate:+.2f}pp")
print(f"辩论者投票偏置诊断: 正方投 pro {pro_votes}/{len(samples)}，反方投 pro {con_votes}/{len(samples)}")
print("（对比 1.5B 中立化：正反方几乎全部投 pro，偏置严重）")
print(f"\n结果已保存到 {OUT_DIR}/irony_debate_7b_*.csv")
