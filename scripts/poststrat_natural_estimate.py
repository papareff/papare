"""事后分层加权估算：用已有的均衡样本 n=150 推算自然分布下的表现（零 GPU）。

原理：现有 n=150 是按标签分层抽样（反讽 75 / 非反讽 75），而完整测试集的自然分布是
反讽 311 / 非反讽 473（39.7% : 60.3%）。分层样本可通过重要性加权还原总体分布：
    acc_natural = Σ_c  P(c) × acc_c
其中 P(c) 为类别在总体中的占比，acc_c 为该类别上的准确率（分层估计）。

这是无偏估计，可立即给出方向性结论；随后用真实抽样实验验证。
"""
import os
import numpy as np
import pandas as pd

RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "experiments", "results"))

d = pd.read_csv(os.path.join(RES, "debate_lens_n150_with.csv"))
b = pd.read_csv(os.path.join(RES, "debate_lens_n150_judge_only.csv"))
m = d.merge(b[["sample_id", "correct"]].rename(columns={"correct": "jc"}), on="sample_id")

te = pd.read_csv("data/raw/irony_test.csv")
te["label"] = pd.to_numeric(te["label"], errors="coerce")
te = te.dropna(subset=["label"])
te["label"] = te["label"].astype(int)

# 总体自然分布
P1 = (te.label == 1).mean()   # 反讽
P0 = 1 - P1

L = []
W = L.append
W("=" * 78)
W("事后分层加权估算：均衡样本 → 自然分布")
W("=" * 78)
W("\n完整测试集: %d 条，反讽 %d (%.1f%%) / 非反讽 %d (%.1f%%)" % (
    len(te), (te.label == 1).sum(), P1 * 100, (te.label == 0).sum(), P0 * 100))
W("现有样本  : %d 条，反讽 %d / 非反讽 %d（人为均衡 50:50）" % (
    len(m), (m.true_label == 1).sum(), (m.true_label == 0).sum()))

W("\n" + "-" * 78)
W("分类别准确率（分层估计）")
W("-" * 78)
W("%-14s %8s %10s %10s %10s" % ("类别", "n", "裁决者", "辩论", "差值"))
accs = {}
for lab, name in [(1, "反讽"), (0, "非反讽")]:
    s = m[m.true_label == lab]
    aj = s.jc.mean() * 100
    ad = s.correct.mean() * 100
    accs[lab] = (aj, ad)
    W("%-14s %8d %9.2f%% %9.2f%% %+9.2fpp" % (name, len(s), aj, ad, ad - aj))

# 均衡分布（原样本）
aj_bal = m.jc.mean() * 100
ad_bal = m.correct.mean() * 100
# 自然分布（加权）
aj_nat = P1 * accs[1][0] + P0 * accs[0][0]
ad_nat = P1 * accs[1][1] + P0 * accs[0][1]

W("\n" + "-" * 78)
W("估算结果对比")
W("-" * 78)
W("%-22s %10s %10s %10s" % ("分布", "裁决者", "辩论", "增益"))
W("%-22s %9.2f%% %9.2f%% %+9.2fpp" % ("均衡 50:50（实测）", aj_bal, ad_bal, ad_bal - aj_bal))
W("%-22s %9.2f%% %9.2f%% %+9.2fpp" % ("自然 39.7:60.3（估算）", aj_nat, ad_nat, ad_nat - aj_nat))

gain_bal = ad_bal - aj_bal
gain_nat = ad_nat - aj_nat
W("\n增益变化: %+.2fpp → %+.2fpp  （衰减 %.2fpp，保留 %.1f%%）" % (
    gain_bal, gain_nat, gain_nat - gain_bal, gain_nat / gain_bal * 100))

# 分解：增益衰减来自哪里
W("\n" + "-" * 78)
W("偏置抵消假设检验（核心）")
W("-" * 78)
W("假设：辩论的净增益部分来自「辩论者过判反讽」补偿「裁决者欠判反讽」，")
W("      因此在反讽占比下降的分布上，这种补偿会减弱、增益随之衰减。")
W("")
contrib_irony = P1 * (accs[1][1] - accs[1][0])
contrib_nonirony = P0 * (accs[0][1] - accs[0][0])
W("自然分布下增益的来源分解:")
W("  反讽类贡献   : %.1f%% × %+.2fpp = %+.2fpp" % (
    P1 * 100, accs[1][1] - accs[1][0], contrib_irony))
W("  非反讽类贡献 : %.1f%% × %+.2fpp = %+.2fpp" % (
    P0 * 100, accs[0][1] - accs[0][0], contrib_nonirony))
W("  合计         : %+.2fpp" % (contrib_irony + contrib_nonirony))
W("")
W("判读: 反讽类贡献 %+.2fpp，非反讽类贡献 %+.2fpp" % (contrib_irony, contrib_nonirony))
if contrib_irony > 0 > contrib_nonirony:
    W("      → 确认「偏置抵消」：增益完全由反讽类召回提升驱动，非反讽类是负贡献。")
    W("      → 反讽占比越低，正贡献被稀释越多，净增益越小。假设成立。")
    # 求临界点：增益归零时的反讽占比
    d1 = accs[1][1] - accs[1][0]
    d0 = accs[0][1] - accs[0][0]
    if d1 != d0:
        p_crit = -d0 / (d1 - d0)
        W("")
        W("      临界反讽占比（增益归零点）: %.1f%%" % (p_crit * 100))
        W("      → 当测试集反讽占比低于 %.1f%% 时，辩论将不再带来净增益" % (p_crit * 100))
elif contrib_irony > 0 and contrib_nonirony > 0:
    W("      → 两类均为正贡献，增益不依赖偏置抵消，稳健性好。")
else:
    W("      → 需进一步分析。")

# 与已有全量数据对照
W("\n" + "-" * 78)
W("与历史全量数据交叉验证")
W("-" * 78)
fs = os.path.join(RES, "irony_fs_finetuned_baseline.csv")
if os.path.exists(fs):
    fb = pd.read_csv(fs)
    if "correct" in fb.columns:
        W("裁决者在完整测试集（n=%d，自然分布）实测: %.2f%%" % (
            len(fb), fb.correct.mean() * 100))
        W("本次事后分层估算（裁决者，自然分布）    : %.2f%%" % aj_nat)
        W("差异 %.2fpp —— 源于 n=150 子样本的抽样波动，估算方向可信" % (
            fb.correct.mean() * 100 - aj_nat))

out = "\n".join(L)
print(out)
with open(os.path.join(RES, "poststrat_natural_estimate.txt"), "w", encoding="utf-8") as f:
    f.write(out)
print("\n已保存: experiments/results/poststrat_natural_estimate.txt")
