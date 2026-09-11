"""诊断视角分化后的反向偏置来源（离线，零 GPU）。

核心问题：修复后判决 pro 占比仅 38%，而真实反讽占比 60%，出现"字面优先"反向偏置。
需定位：是独立判断阶段就偏 con，还是交叉质询阶段被 con 说服？
"""
import os
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "results")
RES = os.path.normpath(RES)
df = pd.read_csv(os.path.join(RES, "fixcheck_neutral50.csv"))

out = []
W = out.append
n = len(df)
gold_pro_rate = (df["true_label"] == 1).mean() * 100

W("=" * 70)
W("反向偏置诊断（视角分化修复后，n=%d）" % n)
W("=" * 70)
W("测试集真实反讽(pro)占比 : %.1f%%" % gold_pro_rate)
W("")

# 1) 各角色投 pro 的比例
W("【各角色投 pro(反讽) 的比例】")
W("  正方(反讽检测视角) pro 票: %.1f%%" % ((df.pro_vote == "pro").mean() * 100))
W("  反方(字面优先视角) pro 票: %.1f%%" % ((df.con_vote == "pro").mean() * 100))
W("  裁决者(DistilBERT)  pro 票: %.1f%%" % ((df.judge_vote == "pro").mean() * 100))
W("  最终判决            pro 票: %.1f%%" % ((df.predicted == "pro").mean() * 100))
W("")

# 2) 分类别召回：辩论对真反讽 vs 真非反讽各自的命中率
W("【分类别表现：辩论是否系统性漏判反讽】")
for lab, name in [(1, "真反讽(应判pro)"), (0, "真非反讽(应判con)")]:
    sub = df[df.true_label == lab]
    hit = (sub.predicted == sub.gold).mean() * 100
    judge_hit = (sub.judge_vote == sub.gold).mean() * 100
    W("  %-16s n=%2d  辩论命中 %.1f%%   裁决者命中 %.1f%%" % (name, len(sub), hit, judge_hit))
W("")

# 3) 投票组合分布：定位偏置发生在哪一步
W("【投票组合分布】（pro_vote/con_vote → 条数）")
combo = df.groupby(["pro_vote", "con_vote"], dropna=False).size()
for (pv, cv), cnt in combo.items():
    W("  pro=%s / con=%s : %d 条" % (pv, cv, cnt))
W("")

# 4) 关键：双方独立判断 vs 最终判决的一致性（质询是否引入了额外偏移）
#    注意：CSV 记录的 pro_vote/con_vote 是质询后的最终投票
W("【质询后双方仍同投 con 的样本，真值分布】")
both_con = df[(df.pro_vote == "con") & (df.con_vote == "con")]
if len(both_con):
    W("  同投 con: %d 条，其中真值是反讽(被漏判)的有 %d 条 (%.1f%%)" % (
        len(both_con), (both_con.true_label == 1).sum(),
        (both_con.true_label == 1).mean() * 100))
both_pro = df[(df.pro_vote == "pro") & (df.con_vote == "pro")]
if len(both_pro):
    W("  同投 pro: %d 条，其中真值是非反讽(被误判)的有 %d 条 (%.1f%%)" % (
        len(both_pro), (both_pro.true_label == 0).sum(),
        (both_pro.true_label == 0).mean() * 100))
W("")

# 5) 门控视角：若用 τ 门控，反向偏置是否仍能被门控规避
W("【门控复核（在修复后的 n=50 上）】")
W("  说明：门控=高置信直接用裁决者，低置信才用辩论")
best = None
for tau in [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]:
    low = df.judge_confidence < tau
    pred = df.predicted.where(low, df.judge_vote)
    acc = (pred == df.gold).mean() * 100
    trig = low.mean() * 100
    if best is None or acc > best[1]:
        best = (tau, acc, trig)
    W("  τ=%.2f: 准确率 %.2f%%  触发率 %.1f%%" % (tau, acc, trig))
judge_acc = (df.judge_vote == df.gold).mean() * 100
debate_acc = (df.predicted == df.gold).mean() * 100
W("")
W("  纯裁决者: %.2f%%   全程辩论: %.2f%%   门控最优(τ=%.2f): %.2f%%" % (
    judge_acc, debate_acc, best[0], best[1]))
W("  门控 vs 纯裁决者: %+.2fpp" % (best[1] - judge_acc))

txt = "\n".join(out)
print(txt)
with open(os.path.join(RES, "diagnose_reverse_bias.txt"), "w", encoding="utf-8") as f:
    f.write(txt)
print("\n已保存: experiments/results/diagnose_reverse_bias.txt")
