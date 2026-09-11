"""n=150 复核结果的真实性核查（纯离线，零 GPU）。

排查四类潜在假象：
  A 类别偏置：是否仍偏向某一类（靠多数类蒙对）
  B 常数预测：判决分布是否退化为常数（预测熵=0）
  C 翻转质量：增益是"改对"还是"改错"，净收益多少条
  D 投票独立性：双方是否真正独立判断，还是同模型重复采样
  E 高置信带核查：增益为 0 的两带是否真的一条未翻转
"""
import math
import numpy as np
import pandas as pd
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "experiments", "results")
RES = os.path.normpath(RES)

d = pd.read_csv(os.path.join(RES, "debate_lens_n150_with.csv"))
b = pd.read_csv(os.path.join(RES, "debate_lens_n150_judge_only.csv"))
m = d.merge(b[["sample_id", "predicted", "correct"]]
            .rename(columns={"predicted": "jp", "correct": "jc"}), on="sample_id")
n = len(m)

L = []
L.append("=" * 74)
L.append("n=150 复核结果真实性核查")
L.append("=" * 74)

# ── A 分类别命中率 ──
L.append("\nA) 分类别命中率（检验是否靠多数类蒙对）")
for lab, name in [(1, "真反讽 pro"), (0, "真非反讽 con")]:
    s = m[m.true_label == lab]
    L.append("   %-14s n=%3d   辩论 %6.2f%%   裁决者 %6.2f%%   差 %+6.2fpp" % (
        name, len(s), s.correct.mean() * 100, s.jc.mean() * 100,
        (s.correct.mean() - s.jc.mean()) * 100))
L.append("   → 判读：若两类别都有正增益，说明不是靠偏置蒙对")

# ── B 常数预测器检验 ──
L.append("\nB) 常数预测器检验")
p_true = (m.true_label == 1).mean() * 100
p_deb = (m.predicted == "pro").mean() * 100
p_judge = (m.jp == "pro").mean() * 100
q = p_deb / 100
ent = -(q * math.log(q) + (1 - q) * math.log(1 - q)) if 0 < q < 1 else 0.0
L.append("   真实反讽占比   : %.2f%%" % p_true)
L.append("   辩论判 pro 率  : %.2f%%" % p_deb)
L.append("   裁决者判 pro 率: %.2f%%" % p_judge)
L.append("   辩论预测熵     : %.4f  (常数预测器=0，最大熵=0.6931)" % ent)
L.append("   与真实占比之差 : %+.2fpp" % (p_deb - p_true))
L.append("   → 判读：熵接近 0 或 pro 率≈真实占比 都说明退化；两者都不成立才是真分类器")

# ── C 翻转质量分析 ──
L.append("\nC) 翻转质量分析（增益的来源）")
fl = m[m.predicted != m.jp]
good = int(((fl.jc == 0) & (fl.correct == 1)).sum())
bad = int(((fl.jc == 1) & (fl.correct == 0)).sum())
L.append("   总翻转条数         : %d / %d" % (len(fl), n))
L.append("   改对（裁决者错→辩论对）: %d 条" % good)
L.append("   改错（裁决者对→辩论错）: %d 条" % bad)
L.append("   净收益             : %+d 条 = %+.2fpp" % (good - bad, (good - bad) / n * 100))
L.append("   改对/改错比        : %.2f : 1" % (good / bad if bad else float("inf")))
L.append("   → 判读：净收益为正且改对显著多于改错，才是真实的纠错能力")

# ── D 投票独立性 ──
L.append("\nD) 投票独立性（是否同模型重复采样）")
agree = d.votes_agree.mean() * 100
pro_pos = (d.pro_vote == "pro").mean() * 100
pro_con = (d.con_vote == "pro").mean() * 100
none_p = int(d.pro_vote.isna().sum())
none_c = int(d.con_vote.isna().sum())
L.append("   双方投票一致率 : %.1f%%  (旧坏版本 98.7%%)" % agree)
L.append("   正方投 pro 率  : %.1f%%" % pro_pos)
L.append("   反方投 pro 率  : %.1f%%" % pro_con)
L.append("   视角方向差     : %+.1fpp (正方=反讽检测视角, 应为正)" % (pro_pos - pro_con))
L.append("   投票解析失败   : 正方 %d 条, 反方 %d 条" % (none_p, none_c))
L.append("   → 判读：一致率在 50~85%% 且方向差不倒挂，才是有效的多视角辩论")

# ── E 高置信带核查 ──
L.append("\nE) 高置信带核查（增益为 0 是否真实）")
for lo, hi, lab in [(0.70, 0.80, "0.70-0.80"), (0.80, 1.01, ">=0.80")]:
    s = m[(m.judge_confidence >= lo) & (m.judge_confidence < hi)]
    flip = int((s.predicted != s.jp).sum())
    L.append("   %-10s n=%3d  翻转 %2d 条  裁决者 %.2f%%  辩论 %.2f%%  差 %+.2fpp" % (
        lab, len(s), flip, s.jc.mean() * 100, s.correct.mean() * 100,
        (s.correct.mean() - s.jc.mean()) * 100))
L.append("   → 判读：若高置信带一条未翻转，说明辩论在高置信区被门控自然屏蔽")

# ── F 统计显著性 ──
L.append("\nF) 统计显著性（配对样本，n=150）")
a1 = m.correct.mean() * 100
a0 = m.jc.mean() * 100
delta = a1 - a0
# 配对差异的标准误：只看翻转样本
disc = m[m.correct != m.jc]
if len(disc):
    b_cnt = int((disc.correct == 1).sum())   # 辩论对裁决者错
    c_cnt = int((disc.jc == 1).sum())        # 裁决者对辩论错
    # McNemar 检验统计量（连续性校正）
    chi2 = (abs(b_cnt - c_cnt) - 1) ** 2 / (b_cnt + c_cnt) if (b_cnt + c_cnt) else 0.0
    se_pair = math.sqrt((b_cnt + c_cnt) - (b_cnt - c_cnt) ** 2 / (b_cnt + c_cnt)) / n * 100 \
        if (b_cnt + c_cnt) else 0.0
else:
    b_cnt = c_cnt = 0
    chi2 = 0.0
    se_pair = 0.0
L.append("   辩论   %.2f%%  vs  裁决者 %.2f%%  →  增益 %+.2fpp" % (a1, a0, delta))
L.append("   不一致对: 辩论对 %d 条 / 裁决者对 %d 条" % (b_cnt, c_cnt))
L.append("   McNemar χ²(连续性校正) = %.3f  → %s" % (
    chi2, "显著 (χ²>3.841, p<0.05)" if chi2 > 3.841 else "不显著"))
L.append("   配对标准误 ≈ %.2fpp，95%%CI = [%+.2f, %+.2f] pp" % (
    se_pair, delta - 1.96 * se_pair, delta + 1.96 * se_pair))

# ── G 门控收益核查 ──
L.append("\nG) 门控收益核查")
g = pd.read_csv(os.path.join(RES, "debate_lens_gate_scan.csv"))
best = g.loc[g.accuracy.idxmax()]
L.append("   最优 τ=%.2f → 准确率 %.2f%%, 触发率 %.1f%%, 延迟 %.2fs" % (
    best.tau, best.accuracy, best.trigger_rate, best.weighted_latency))
L.append("   纯裁决者      59.33%%   延迟 0.01s")
L.append("   全程辩论      %.2f%%   延迟 %.2fs" % (a1, d.latency_sec.mean()))
L.append("   门控 τ=%.2f    %.2f%%   延迟 %.2fs" % (best.tau, best.accuracy, best.weighted_latency))
L.append("   → 门控 vs 纯裁决者: %+.2fpp, 延迟 %.0f 倍" % (
    best.accuracy - a0, best.weighted_latency / 0.01))
L.append("   → 门控 vs 全程辩论: %+.2fpp, 节省延迟 %.0f%%" % (
    best.accuracy - a1, (1 - best.weighted_latency / d.latency_sec.mean()) * 100))
L.append("   ⚠ 注意：门控触发率 %.0f%% 已接近全程辩论，成本优势大幅削弱" % best.trigger_rate)

out = "\n".join(L)
print(out)
with open(os.path.join(RES, "verify_n150_audit.txt"), "w", encoding="utf-8") as f:
    f.write(out)
print("\n已保存: experiments/results/verify_n150_audit.txt")
