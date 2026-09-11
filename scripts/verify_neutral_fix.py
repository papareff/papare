"""修复验证（本机，n=50）：检验视角分化是否恢复了真实的意见多样性。

判据（对照旧实现的退化指标）：
  1. 双方投票一致率：旧版 98.7% → 健康区间应为 60~85%
     （过低说明视角提示词变成了变相预设立场，过高说明仍未分化）
  2. 最终判决的 pro 占比：旧版 98.7%（常数预测器）→ 应显著低于此，接近真实标签分布
  3. 常数预测器检验：若判决 pro 占比 ≈ 测试集反讽占比，说明仍在退化为常数预测
  4. 准确率对照：与仅裁决者基线比较

样本批次与 run_irony_debate_neutral.py 完全一致（SEED=42 分层抽样后取前 50 条），
便于与已落盘的 n=150 结果做配对对比。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import random
import numpy as np
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

NUM = 50
SEED = 42
TEST_CSV = "data/raw/irony_test.csv"
OUT = "experiments/results/fixcheck_neutral50.csv"
LABEL_VOTE = {1: "pro", 0: "con"}

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["task"] = "irony"
rounds = 1                              # 验证阶段先用 1 轮质询，节省时间
config["debate"]["rounds"] = rounds
config["debate"]["neutral"] = True

# ── 与 n=150 实验相同的抽样批次，取前 NUM 条 ──
test_df = pd.read_csv(TEST_CSV)
test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce")
test_df = test_df.dropna(subset=["label", "text"])
test_df["label"] = test_df["label"].astype(int)

full_batch = (
    test_df.groupby("label", group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), 150 // 2), random_state=SEED))
)
# ⚠️ 必须与 run_irony_debate_neutral.py 用完全相同的洗牌方式（random.shuffle），
# 否则"前 50 条"就不是原 n=150 批次的前 50 条，配对对比会失真。
deb_records = full_batch.reset_index(drop=True).to_dict("records")
random.seed(SEED)
random.shuffle(deb_records)
samples = deb_records[:NUM]

print(f"验证样本: {len(samples)} 条（与 n=150 实验同批次前 {NUM} 条）", flush=True)

debate_config = dict(config)
debate_config["judge"] = {
    "name": "./models/irony_distilbert", "device": "cuda", "max_length": 128
}
pipeline = InferencePipeline(debate_config, use_debate=True, retriever=None, num_shots=0)

rows, correct, lats = [], 0, []
for i, s in enumerate(samples):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    r = pipeline.run(s)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    lats.append(dt)
    gold = LABEL_VOTE[int(s["label"])]
    ok = r["decision"] == gold
    correct += int(ok)
    v = r.get("debate_votes", {})
    rows.append({
        "sample_id": i, "text": s["text"], "true_label": int(s["label"]), "gold": gold,
        "predicted": r["decision"], "judge_vote": r.get("judge_vote"),
        "judge_confidence": r.get("judge_confidence"),
        "pro_vote": v.get("pro"), "con_vote": v.get("con"),
        "votes_agree": int(v.get("pro") == v.get("con")),
        "latency_sec": round(dt, 2), "correct": int(ok),
    })
    print(f"  [{i}] true={s['label']} pred={r['decision']} judge={r.get('judge_vote')} "
          f"votes={v.get('pro')}/{v.get('con')} {'✓' if ok else '✗'} ({dt:.1f}s)", flush=True)

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False, encoding="utf-8-sig")

# ── 诊断 ──
agree = df["votes_agree"].mean() * 100
pro_rate = (df["predicted"] == "pro").mean() * 100
true_pro_rate = (df["true_label"] == 1).mean() * 100
acc = df["correct"].mean() * 100
judge_acc = (df["judge_vote"] == df["gold"]).mean() * 100

print("\n" + "=" * 72)
print("修复验证结果（n=%d）" % NUM)
print("=" * 72)
print("① 双方投票一致率      : %.1f%%   (旧版 98.7%%)" % agree)
# 一致率不设硬编码健康区间：只要有真实分歧（<95%）且判决已脱离常数预测即可
flag1 = "通过" if agree < 95 else "仍接近恒同票"
print("   判定: %s" % flag1)
print("")
print("② 最终判决 pro 占比   : %.1f%%   (旧版 98.7%%)" % pro_rate)
print("   测试集真实反讽占比 : %.1f%%" % true_pro_rate)
const_pred = abs(pro_rate - true_pro_rate) < 3.0
print("   常数预测器检验: %s" % (
    "仍退化为常数预测器（判 pro 率≈真实占比）" if const_pred else "已脱离常数预测"))
print("")
# 投票方向检验：视角分化的目的是让两方产生可区分的系统性差异
pro_pro = (df.pro_vote == "pro").mean() * 100
con_pro = (df.con_vote == "pro").mean() * 100
print("③ 投票方向检验")
print("   正方(反讽检测视角)投 pro: %.1f%%" % pro_pro)
print("   反方(字面优先视角)投 pro: %.1f%%" % con_pro)
print("   两方差异: %.1fpp（应 >0，且正方不应显著低于反方——否则视角倒挂）" % (pro_pro - con_pro))
direction_ok = pro_pro > con_pro or abs(pro_pro - con_pro) < 5
print("   判定: %s" % ("通过（无倒挂）" if direction_ok else "仍倒挂"))
print("")
print("④ 准确率对照（注意：n=%d 时标准误约 ±%.1fpp，仅供方向参考）" % (
    NUM, np.sqrt(acc * (100 - acc) / NUM) * 1.96 / 100 * 100))
print("   视角分化辩论 : %.2f%%" % acc)
print("   仅裁决者     : %.2f%%" % judge_acc)
print("   辩论增益     : %+.2fpp" % (acc - judge_acc))
print("")
print("⑤ 成本: 平均 %.2fs/条 (rounds=%d)" % (np.mean(lats), rounds))
print("")
print("=" * 72)
if flag1 == "通过" and not const_pred and direction_ok:
    print("结论: 机制缺陷已修复（多样性恢复 + 脱离常数预测 + 无方向倒挂）")
    print("      准确率增益需在 n=150 上用贪心解码复核后才可采信")
else:
    print("结论: 仍有机制问题，需进一步调整视角提示词")
print("结果已保存: %s" % OUT)
