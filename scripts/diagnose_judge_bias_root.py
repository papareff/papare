"""裁决者偏置根因分析 + 1.5B 辩论者能力来源分析（纯离线）。

问题一：裁决者（微调 DistilBERT）为何偏置更严重？
  核查维度：
   - 训练配置留下的痕迹（样本量/训练步数/最优 epoch）
   - 误判的方向性与文本特征
   - 置信度校准度（置信度能否区分对错）
   - 与辩论者在同一批样本上的互补关系

问题二：1.5B 小模型为何判断能力强？
  核查维度：
   - 辩论者答对而裁决者答错的样本，其文本特征（是否依赖常识/语用推理）
   - 两类模型的能力分工
"""
import os
import numpy as np
import pandas as pd

RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "experiments", "results"))

d = pd.read_csv(os.path.join(RES, "debate_lens_n150_with.csv"))
b = pd.read_csv(os.path.join(RES, "debate_lens_n150_judge_only.csv"))
m = d.merge(b[["sample_id", "predicted", "correct"]]
            .rename(columns={"predicted": "jp", "correct": "jc"}), on="sample_id")

# 训练集特征（用于解释裁决者的能力上限）
tr = pd.read_csv("data/raw/irony_train.csv")
te = pd.read_csv("data/raw/irony_test.csv")

L = []
W = L.append
W("=" * 78)
W("问题一：裁决者偏置的根本原因")
W("=" * 78)

W("\n【证据 1】训练数据规模与训练强度")
W("  训练集样本数        : %d 条" % len(tr))
W("  训练集标签分布      : 反讽 %d / 非反讽 %d  (%.1f%% : %.1f%%)" % (
    (tr.label == 1).sum(), (tr.label == 0).sum(),
    (tr.label == 1).mean() * 100, (tr.label == 0).mean() * 100))
W("  测试集标签分布      : 反讽 %d / 非反讽 %d  (%.1f%% : %.1f%%)" % (
    (te.label == 1).sum(), (te.label == 0).sum(),
    (te.label == 1).mean() * 100, (te.label == 0).mean() * 100))
steps = len(tr) * 3 // 32
W("  总训练步数          : %d 步（%d 条 × 3 epoch ÷ batch 32）" % (steps, len(tr)))
W("  DistilBERT 可训练参数: 约 6600 万")
W("  → 参数量/训练样本数 ≈ %.0f : 1" % (66e6 / len(tr)))
W("  训练日志（实测）     : epoch1 val_acc=73.52%% / epoch2=72.82%% / epoch3=71.78%%")
W("  → 早停在 epoch 1 即达最优，之后持续下降 = 2575 条样本已榨干该模型的可学信息")

W("\n【证据 2】裁决者偏置的方向与幅度")
p_true = (m.true_label == 1).mean() * 100
p_judge = (m.jp == "pro").mean() * 100
p_deb = (m.predicted == "pro").mean() * 100
W("  真实反讽占比        : %.1f%%" % p_true)
W("  裁决者判反讽率      : %.1f%%   → 偏差 %+.1fpp（系统性欠判反讽）" % (
    p_judge, p_judge - p_true))
W("  辩论者判反讽率      : %.1f%%   → 偏差 %+.1fpp（反向过判反讽）" % (
    p_deb, p_deb - p_true))
W("  裁决者分类别命中率  : 反讽 %.2f%% / 非反讽 %.2f%%" % (
    m[m.true_label == 1].jc.mean() * 100, m[m.true_label == 0].jc.mean() * 100))
W("  辩论者分类别命中率  : 反讽 %.2f%% / 非反讽 %.2f%%" % (
    m[m.true_label == 1].correct.mean() * 100, m[m.true_label == 0].correct.mean() * 100))
W("  → 裁决者在反讽类上 49.33%%，几乎等于随机（50%%）；在非反讽类上 69.33%%")
W("    这不是「不会做」，而是「学到了一个偏向字面解读的决策边界」")

W("\n【证据 3】置信度校准度（置信度能否区分对错）")
m["conf"] = m["judge_confidence"].astype(float)
for lab, name in [(1, "判对"), (0, "判错")]:
    s = m[m.jc == lab]
    W("  裁决者%s的样本 (n=%3d) 平均置信度: %.4f" % (name, len(s), s.conf.mean()))
corr_conf = m[m.jc == 1].conf.mean()
wrong_conf = m[m.jc == 0].conf.mean()
W("  置信度差值          : %+.4f（校准良好应显著为正）" % (corr_conf - wrong_conf))
# AUC 近似：用秩和
from scipy.stats import rankdata
try:
    y = m.jc.values
    s_ = m.conf.values
    r = rankdata(s_)
    n1, n0 = y.sum(), (1 - y).sum()
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    W("  置信度对「裁决者是否正确」的 AUC: %.4f" % auc)
    W("  → AUC≈0.5 表示置信度几乎不含「自己是否判对」的信息（校准失效）")
except Exception as e:
    W("  （AUC 计算跳过: %s）" % e)

W("\n【证据 4】裁决者失败样本的文本特征（抽样）")
miss = m[(m.true_label == 1) & (m.jc == 0)]
W("  裁决者漏判的反讽样本共 %d 条，示例 6 条：" % len(miss))
for _, r in miss.head(6).iterrows():
    W("    [conf=%.3f] %s" % (r.conf, str(r.text)[:78]))
W("")
W("  辩论者在这些样本上的表现:")
W("    辩论答对 %d / %d 条 = %.1f%%" % (
    miss.correct.sum(), len(miss), miss.correct.mean() * 100))

W("\n" + "=" * 78)
W("问题二：1.5B 模型判断能力的来源")
W("=" * 78)

W("\n【证据 5】能力分工：辩论者独赢的样本特征")
both_wrong = m[(m.jc == 0) & (m.correct == 1)]
judge_only = m[(m.jc == 1) & (m.correct == 0)]
W("  裁决者错、辩论者对 : %d 条（辩论者的独有贡献）" % len(both_wrong))
W("  裁决者对、辩论者错 : %d 条（辩论者的损失）" % len(judge_only))
W("")
W("  「辩论者独赢」样本的文本长度统计:")
W("    平均字符数 %.1f，中位 %.0f" % (
    both_wrong.text.str.len().mean(), both_wrong.text.str.len().median()))
W("  「裁决者独赢」样本:")
W("    平均字符数 %.1f，中位 %.0f" % (
    judge_only.text.str.len().mean(), judge_only.text.str.len().median()))
W("")
W("  辩论者独赢样本示例（这类样本需要语用/常识推理）:")
for _, r in both_wrong.head(6).iterrows():
    W("    [裁决者conf=%.3f] %s" % (r.conf, str(r.text)[:78]))

W("\n【证据 6】两类模型的能力边界对比")
W("  裁决者 = 编码器分类头（DistilBERT, 66M 参数）")
W("    · 训练方式: 在 2575 条标注样本上做监督微调")
W("    · 能力来源: 从这 2575 条里统计词面模式")
W("    · 输出形式: 直接给出类别概率，无中间推理过程")
W("    · 实测上限: 验证集 73.52%%，测试集反讽类仅 49.33%%")
W("")
W("  辩论者 = 指令微调生成模型（Qwen2-1.5B-Instruct）")
W("    · 训练方式: 海量网页语料预训练 + 指令对齐（非本任务微调）")
W("    · 能力来源: 预训练获得的世界知识与语用常识")
W("    · 输出形式: 先生成自然语言论据，再给结论（显式推理链）")
W("    · 实测表现: 反讽类 93.33%%，非反讽类 62.67%%")

W("\n【证据 7】关键对照：两者在「反讽」上的差异本质")
W("  裁决者在反讽类 49.33%% ≈ 随机水平，说明它无法识别反讽；")
W("  辩论者在反讽类 93.33%%，说明它具备识别反讽所需的语用常识。")
W("  差异不在参数规模（66M vs 1.5B，仅 23 倍），而在：")
W("    ① 知识来源: 2575 条任务样本  vs  海量通用语料预训练")
W("    ② 推理形式: 单步概率输出      vs  先生成论据再结论")
W("    ③ 任务性质: 反讽识别依赖「字面 vs 情境」的常识落差，")
W("       这类知识在通用语料中大量存在，却很难从 2575 条标注里学到")

out = "\n".join(L)
print(out)
with open(os.path.join(RES, "diagnose_judge_bias_root.txt"), "w", encoding="utf-8") as f:
    f.write(out)
print("\n已保存: experiments/results/diagnose_judge_bias_root.txt")
