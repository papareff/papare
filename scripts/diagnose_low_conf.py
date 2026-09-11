"""诊断：低置信带内裁决者准确率为何异常低/波动大。

检验三个候选解释：
H1 标签构成差异：带内真值标签比例不同，若裁决者有类别偏置，则准确率随构成波动
H2 裁决者类别偏置：裁决者系统性偏向预测某一类（如全猜 con）
H3 小样本随机波动：conf≈0.5 时模型本无区分力，准确率应在 50% 附近随机抖动
"""
import os
import math
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "experiments", "results")
RES = os.path.normpath(RES)

df = pd.read_csv(os.path.join(RES, "irony_debate_neutral_without.csv"))
df["conf"] = df["judge_confidence"].astype(float)
# pro = 反讽(1)，con = 非反讽(0)
df["pred_num"] = df["predicted"].map({"pro": 1, "con": 0})

out = []
W = out.append

W("=" * 78)
W("【整体】裁决者（无辩论基线，n=%d）" % len(df))
W("=" * 78)
acc = df["correct"].mean()
W("  总准确率           : %.2f%%" % (acc * 100))
W("  真值构成           : 反讽(1)=%d  非反讽(0)=%d" % (
    (df.true_label == 1).sum(), (df.true_label == 0).sum()))
W("  裁决者预测构成     : pro=%d  con=%d   → 预测偏置指数 %.3f" % (
    (df.pred_num == 1).sum(), (df.pred_num == 0).sum(),
    (df.pred_num == 1).mean()))
W("")
W("  分类别表现（裁决者是否存在类别偏置 = H2 检验）:")
for lab, name in [(1, "真值=反讽(pro)"), (0, "真值=非反讽(con)")]:
    sub = df[df.true_label == lab]
    W("    %-16s n=%3d  裁决者准确率 %.2f%%   其中预测为 pro 的占比 %.1f%%" % (
        name, len(sub), sub["correct"].mean() * 100,
        (sub.pred_num == 1).mean() * 100))
W("")

# ── H3：二项检验，判断各带准确率是否显著偏离 50%（随机水平）──
def binom_p_two_sided(k, n, p=0.5):
    """双侧二项检验 p 值（正态近似 + 连续性校正）"""
    if n == 0:
        return float("nan")
    mean, sd = n * p, math.sqrt(n * p * (1 - p))
    if sd == 0:
        return float("nan")
    z = (abs(k - mean) - 0.5) / sd
    # 标准正态双侧
    return math.erfc(abs(z) / math.sqrt(2))


W("=" * 78)
W("【分带诊断】H1 标签构成 + H3 随机波动检验")
W("=" * 78)
W("%-13s %4s %7s %7s %9s %9s %8s %9s" % (
    "置信带", "n", "裁决acc", "真反讽%", "预pro%", "对pro-acc", "对con-acc", "p(≠50%)"))
W("-" * 78)

bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
rows = []
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    sub = df[(df.conf >= lo) & (df.conf < hi)]
    if len(sub) == 0:
        continue
    n = len(sub)
    k = int(sub["correct"].sum())
    acc_b = k / n
    pro_true = (sub.true_label == 1).mean() * 100
    pred_pro = (sub.pred_num == 1).mean() * 100
    s_pro = sub[sub.true_label == 1]
    s_con = sub[sub.true_label == 0]
    acc_pro = s_pro["correct"].mean() * 100 if len(s_pro) else float("nan")
    acc_con = s_con["correct"].mean() * 100 if len(s_con) else float("nan")
    pv = binom_p_two_sided(k, n)
    W("%-13s %4d %6.2f%% %6.1f%% %8.1f%% %8s%% %8s%% %8.3f" % (
        "[%.2f,%.2f)" % (lo, hi), n, acc_b * 100, pro_true, pred_pro,
        ("%.2f" % acc_pro) if acc_pro == acc_pro else "—",
        ("%.2f" % acc_con) if acc_con == acc_con else "—",
        pv))
    rows.append(dict(band="%.2f-%.2f" % (lo, hi), n=n, acc=acc_b * 100,
                     pro_true=pro_true, pred_pro=pred_pro, p=pv))

W("")
W("=" * 78)
W("【两个反常带的对照】[0.50,0.55) acc=64% vs [0.55,0.60) acc=41%")
W("=" * 78)
b1 = df[(df.conf >= 0.50) & (df.conf < 0.55)]
b2 = df[(df.conf >= 0.55) & (df.conf < 0.60)]
for name, b in [("[0.50,0.55)", b1), ("[0.55,0.60)", b2)]:
    W("  %s  n=%d" % (name, len(b)))
    W("     真值构成 : 反讽=%d 非反讽=%d  (反讽占 %.1f%%)" % (
        (b.true_label == 1).sum(), (b.true_label == 0).sum(),
        (b.true_label == 1).mean() * 100))
    W("     预测构成 : pro=%d con=%d" % ((b.pred_num == 1).sum(), (b.pred_num == 0).sum()))
    W("     裁决 acc : %.2f%%   二项检验 p(vs 50%%) = %.3f" % (
        b["correct"].mean() * 100,
        binom_p_two_sided(int(b["correct"].sum()), len(b))))
    W("     平均置信 : %.4f   置信范围 [%.4f, %.4f]" % (
        b.conf.mean(), b.conf.min(), b.conf.max()))
    W("")

# ── 合并两个带后的整体水平 ──
both = pd.concat([b1, b2])
W("  两带合并（n=%d）: 裁决 acc = %.2f%%，p(vs 50%%) = %.3f" % (
    len(both), both["correct"].mean() * 100,
    binom_p_two_sided(int(both["correct"].sum()), len(both))))
W("")

# ── 置信度与正确率的相关性 ──
r = df["conf"].corr(df["correct"].astype(float))
W("=" * 78)
W("【置信度校准性】")
W("=" * 78)
W("  置信度 与 是否正确 的 Pearson 相关系数 r = %.4f" % r)
lo = df[df.conf < 0.6]
hi = df[df.conf >= 0.8]
W("  低置信组 (conf<0.6)  n=%3d  acc=%.2f%%  平均置信=%.4f" % (
    len(lo), lo["correct"].mean() * 100, lo.conf.mean()))
W("  高置信组 (conf>=0.8) n=%3d  acc=%.2f%%  平均置信=%.4f" % (
    len(hi), hi["correct"].mean() * 100, hi.conf.mean()))
W("")
W("  → 若校准良好，高置信组 acc 应显著高于低置信组；差值 = %+.2fpp" % (
    (hi["correct"].mean() - lo["correct"].mean()) * 100))

txt = "\n".join(out)
print(txt)
with open(os.path.join(RES, "diagnose_low_conf_accuracy.txt"), "w", encoding="utf-8") as f:
    f.write(txt)
print("\n已保存: experiments/results/diagnose_low_conf_accuracy.txt")
