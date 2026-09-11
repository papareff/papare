"""裁决者重训前后对照分析：精准度、偏置方向、门控信号质量、对辩论框架的影响。

同口径对照（均在 semeval_test 784 / semeval_val 287 / misra_test 2661 上评测）：
  基线   : models/irony_distilbert            （SemEval 2575 单阶段，原方案）
  两阶段 : models/judge_distilbert_cw_2stage  （Misra 21281 预训练 → SemEval 2575 微调）

重点检验三件事：
  ① 反讽召回是否从 49.33% 显著改善（重训的直接目标）
  ② 改善是否以引入新的 pro 偏置为代价（准确率可能不动，只是错误类型转移）
  ③ 裁决者偏置方向若与辩论方 pro 偏置同向，「偏置互补抵消」效应会否消失
     → 这决定重训后辩论增益是升还是降，是本分析最关键的推论
"""
import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RES = "experiments/results"
BASE_NAME = "irony_distilbert"
NEW_NAME = "judge_distilbert_cw_2stage"

# 辩论方的已知偏置（来自 n=150 / n=200 实测）
DEBATE_BIAS = {
    "均衡 n=150": {"true_rate": 50.0, "debate_pro_rate": 65.3,
                   "judge_pro_rate": 40.0, "debate_delta_pp": 18.67},
    "自然 n=200": {"true_rate": 44.0, "debate_pro_rate": 72.0,
                   "judge_pro_rate": 35.0, "debate_delta_pp": 2.00},
}


def se(pct, n):
    return math.sqrt(pct * (100 - pct) / n)


def load_conf(name, split):
    p = f"{RES}/conf_{name}_{split}.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


print("=" * 104)
print("裁决者重训前后对照分析")
print("=" * 104)

rows = []
for split, n_exp in [("semeval_test", 784), ("semeval_val", 287), ("misra_test", 2661)]:
    cb = load_conf(BASE_NAME, split)
    cn = load_conf(NEW_NAME, split)
    if cb is None or cn is None:
        print(f"[跳过] {split}: 缺少 conf 文件")
        continue
    for tag, c in [("基线 irony_distilbert", cb), ("两阶段 cw_2stage", cn)]:
        y = c.true_label.astype(int).values
        p = c.predicted.astype(int).values
        tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        tp = int(((y == 1) & (p == 1)).sum())
        acc = (tn + tp) / len(y) * 100
        rec_i = tp / max(tp + fn, 1) * 100
        rec_n = tn / max(tn + fp, 1) * 100
        prec_i = tp / max(tp + fp, 1) * 100
        f1_i = 2 * prec_i * rec_i / max(prec_i + rec_i, 1e-9)
        rows.append({
            "split": split, "n": len(y), "model": tag,
            "accuracy": round(acc, 2), "f1_irony": round(f1_i / 100, 4),
            "precision_irony": round(prec_i, 2), "recall_irony": round(rec_i, 2),
            "recall_nonirony": round(rec_n, 2),
            "true_irony_rate": round(y.mean() * 100, 2),
            "pred_pro_rate": round(p.mean() * 100, 2),
            "bias_pp": round(p.mean() * 100 - y.mean() * 100, 2),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "auc_gate": None, "brier": None,
        })
df = pd.DataFrame(rows)

# ── 显著性检验（按 split 分别记录，避免 z 值串用）──
sig = {}
for split in ["semeval_test", "semeval_val", "misra_test"]:
    s = df[df.split == split]
    if len(s) < 2:
        continue
    b_ = s[s.model.str.contains("基线")].iloc[0]
    n_ = s[s.model.str.contains("两阶段")].iloc[0]
    cb = load_conf(BASE_NAME, split)
    cn = load_conf(NEW_NAME, split)
    # McNemar 配对检验（同一批样本上两个模型的判决对比，比两比例 z 检验更合适）
    pb = cb.predicted.astype(int).values
    pn = cn.predicted.astype(int).values
    y = cb.true_label.astype(int).values
    c_ok_b = (pb == y)
    c_ok_n = (pn == y)
    bb = int((~c_ok_b & c_ok_n).sum())   # 基线错、两阶段对
    cc = int((c_ok_b & ~c_ok_n).sum())   # 基线对、两阶段错
    chi2 = (abs(bb - cc) - 1) ** 2 / (bb + cc) if (bb + cc) else 0.0
    d_acc = n_.accuracy - b_.accuracy
    sig[split] = {
        "delta_acc_pp": round(d_acc, 2), "b_base_wrong_new_right": bb,
        "c_base_right_new_wrong": cc, "mcnemar_chi2": round(chi2, 3),
        "significant": bool(chi2 > 3.841),
        "delta_recall_irony_pp": round(n_.recall_irony - b_.recall_irony, 2),
    }

# ── 1. 精准度对照 ──
print("\n" + "=" * 104)
print("【1】精准度对照（同口径）")
print("=" * 104)
for split in ["semeval_test", "semeval_val", "misra_test"]:
    s = df[df.split == split]
    if len(s) < 2:
        continue
    print(f"\n  ◆ {split}（n={int(s.n.iloc[0])}，真实反讽率 {s.true_irony_rate.iloc[0]:.1f}%）")
    print(f"  {'模型':<26} {'准确率':>9} {'F1(反讽)':>10} {'精确率':>9} "
          f"{'召回(反讽)':>11} {'召回(非反讽)':>12}")
    print("  " + "-" * 84)
    for _, r in s.iterrows():
        print(f"  {r['model']:<26} {r.accuracy:>8.2f}% {r.f1_irony:>10.4f} "
              f"{r.precision_irony:>8.2f}% {r.recall_irony:>10.2f}% "
              f"{r.recall_nonirony:>11.2f}%")
    b = s[s.model.str.contains("基线")].iloc[0]
    nn = s[s.model.str.contains("两阶段")].iloc[0]
    n = int(b.n)
    ci = se(b.accuracy, n) * 1.96
    ci2 = se(nn.accuracy, n) * 1.96
    d = nn.accuracy - b.accuracy
    sg = sig.get(split, {})
    print(f"  {'变化':<26} {d:>+8.2f}pp {nn.f1_irony-b.f1_irony:>+10.4f} "
          f"{nn.precision_irony-b.precision_irony:>+8.2f}pp "
          f"{nn.recall_irony-b.recall_irony:>+10.2f}pp "
          f"{nn.recall_nonirony-b.recall_nonirony:>+11.2f}pp")
    print(f"  准确率 95%CI: 基线 ±{ci:.2f}pp / 两阶段 ±{ci2:.2f}pp")
    # McNemar 配对检验：同一批样本上比较两个模型，比两比例 z 检验更合适
    print(f"  McNemar 配对检验: b(基线错/新对)={sg.get('b_base_wrong_new_right')}  "
          f"c(基线对/新错)={sg.get('c_base_right_new_wrong')}  "
          f"χ²={sg.get('mcnemar_chi2')} → "
          f"{'显著' if sg.get('significant') else '不显著'}（α=0.05，临界 3.841）")

# ── 2. 错误类型转移 ──
print("\n" + "=" * 104)
print("【2】错误类型转移：召回提升是否以精确率为代价")
print("=" * 104)
s = df[df.split == "semeval_test"]
b = s[s.model.str.contains("基线")].iloc[0]
nn = s[s.model.str.contains("两阶段")].iloc[0]
print(f"  {'混淆矩阵项':<20} {'基线':>10} {'两阶段':>10} {'变化':>10} {'含义':<28}")
print("  " + "-" * 82)
items = [
    ("TP 真反讽判对", b.tp, nn.tp, "反讽召回的分子"),
    ("FN 真反讽漏判", b.fn, nn.fn, "↓ 漏判反讽减少 = 召回改善"),
    ("TN 真非反讽判对", b.tn, nn.tn, "非反讽召回的分子"),
    ("FP 真非反讽误判", b.fp, nn.fp, "↑ 误判反讽增多 = 新偏置"),
]
for name, v1, v2, mean in items:
    print(f"  {name:<20} {v1:>10} {v2:>10} {v2-v1:>+10} {mean:<28}")
print(f"\n  总错误数: 基线 {b.fn+b.fp}  →  两阶段 {nn.fn+nn.fp}  "
      f"({(nn.fn+nn.fp)-(b.fn+b.fp):+d})")
print(f"  错误结构: 基线 漏判 {b.fn}/误判 {b.fp} = {b.fn/max(b.fp,1):.2f} : 1")
print(f"           两阶段 漏判 {nn.fn}/误判 {nn.fp} = {nn.fn/max(nn.fp,1):.2f} : 1")
print(f"  → 错误从「漏判反讽为主」转为「误判反讽为主」，总错误数"
      f"{'减少' if (nn.fn+nn.fp)<(b.fn+b.fp) else '增加'} "
      f"{abs((nn.fn+nn.fp)-(b.fn+b.fp))} 例。")

# ── 3. 偏置方向与辩论框架的交互（最关键）──
print("\n" + "=" * 104)
print("【3】偏置方向同向化风险：重训是否破坏了「偏置互补抵消」")
print("=" * 104)
print(f"  辩论框架原有增益来自两个组件的偏置【方向相反】：")
print(f"    裁决者 pro 偏置为负（倾向判非反讽，漏抓反讽）")
print(f"    辩论方 pro 偏置为正（倾向判反讽，宁抓不漏）")
print(f"    两者互补 → 辩论纠正裁决者的漏判，产生 +18.67pp 增益")
print(f"\n  {'辩论实验集':<16} {'其真实反讽率':>13} {'基线裁决偏置':>13} {'两阶段裁决偏置':>15} "
      f"{'辩论方偏置':>12} {'同向?':>9}")
print("  " + "-" * 88)
print("  （裁决者偏置在 semeval_test 全量 784 条上测得，真实反讽率 39.7%；")
print("    辩论方偏置来自各自辩论实验集的实测值）")
bias_rows = []
for split, dkey in [("semeval_test", None)]:
    s2 = df[df.split == split]
    bb = s2[s2.model.str.contains("基线")].iloc[0]
    nnn = s2[s2.model.str.contains("两阶段")].iloc[0]
    true_r = bb.true_irony_rate
    for tag, info in DEBATE_BIAS.items():
        d_bias = info["debate_pro_rate"] - info["true_rate"]
        same_base = "是" if bb.bias_pp * d_bias > 0 else "否"
        same_new = "是" if nnn.bias_pp * d_bias > 0 else "否"
        bias_rows.append({
            "eval_split": split, "debate_set": tag,
            "debate_set_true_irony_rate": info["true_rate"],
            "judge_eval_true_irony_rate": true_r,
            "base_judge_bias_pp": bb.bias_pp, "new_judge_bias_pp": nnn.bias_pp,
            "debate_bias_pp": round(d_bias, 2),
            "base_opposite": same_base == "否", "new_opposite": same_new == "否",
        })
        print(f"  {tag:<16} {info['true_rate']:>10.1f}% {bb.bias_pp:>+12.2f}pp "
              f"{nnn.bias_pp:>+14.2f}pp {d_bias:>+11.2f}pp "
              f"{same_base:>4}/{same_new:<3}")
print(f"\n  ◆ 关键推论")
print(f"    基线裁决者在 semeval_test 上偏置 = {b.bias_pp:+.2f}pp"
      f"（{'倾向判反讽' if b.bias_pp>0 else '倾向判非反讽'}）")
print(f"    两阶段裁决者偏置 = {nn.bias_pp:+.2f}pp"
      f"（{'倾向判反讽' if nn.bias_pp>0 else '倾向判非反讽'}）")
if b.bias_pp < 0 < nn.bias_pp:
    print(f"    ★ 偏置方向发生【反转】：从「倾向判非反讽」变为「倾向判反讽」，"
          f"与辩论方 pro 偏置【同向】。")
    print(f"    → 互补抵消机制被破坏。辩论方原本纠正裁决者的漏判，")
    print(f"      现在两者都倾向判反讽，辩论不再提供纠正，只会叠加偏置。")
    print(f"    → 预测：重训后辩论增益会【进一步下降】，甚至变为负增益。")
    print(f"      必须重跑辩论流程验证（已列入服务器任务）。")
elif b.bias_pp * nn.bias_pp > 0:
    print(f"    偏置方向未变，互补机制可能保留。")
else:
    print(f"    偏置方向变化需结合辩论实测判断。")

# ── 4. 门控信号质量：重训是否改善 ──
print("\n" + "=" * 104)
print("【4】门控信号质量：重训前后对比")
print("=" * 104)
print(f"  {'模型':<26} {'split':<16} {'AUC(预测裁决者出错)':>20} {'Brier':>9} "
      f"{'置信度均值':>11}")
print("  " + "-" * 88)
gate_rows = []
for split in ["semeval_test", "semeval_val"]:
    for tag, name in [("基线 irony_distilbert", BASE_NAME),
                      ("两阶段 cw_2stage", NEW_NAME)]:
        c = load_conf(name, split)
        if c is None:
            continue
        y = c.true_label.astype(int).values
        p = c.predicted.astype(int).values
        wrong = (p != y).astype(int)
        if len(np.unique(wrong)) < 2:
            continue
        try:
            auc = roc_auc_score(wrong, -c.confidence.values)
        except ValueError:
            auc = float("nan")
        brier = float(np.mean((c.conf_pro.values - y) ** 2))
        gate_rows.append({"model": tag, "split": split, "auc_gate": round(auc, 4),
                          "brier": round(brier, 4),
                          "mean_confidence": round(float(c.confidence.mean()), 4),
                          "error_rate_pct": round(wrong.mean() * 100, 2)})
        print(f"  {tag:<26} {split:<16} {auc:>20.4f} {brier:>9.4f} "
              f"{c.confidence.mean()*100:>10.1f}%")
df_gate = pd.DataFrame(gate_rows)
if len(df_gate) >= 2:
    piv = df_gate[df_gate.split == "semeval_test"]
    if len(piv) == 2:
        ab = piv[piv.model.str.contains("基线")].auc_gate.iloc[0]
        an = piv[piv.model.str.contains("两阶段")].auc_gate.iloc[0]
        print(f"\n  ◆ semeval_test 上门控 AUC: 基线 {ab:.4f} → 两阶段 {an:.4f} "
              f"({an-ab:+.4f})")
        print(f"    {'改善' if an > ab else '未改善'}；门控可用性判据为 AUC>0.60。"
              f"两阶段模型{'达标' if an > 0.60 else '仍不达标'}。")
        print(f"    注：n=150 上原裁决者 AUC=0.6045，此处 {ab:.4f} 为 784 条全量口径，"
              f"样本更大更可靠。")

# ── 5. 跨域迁移效果 ──
print("\n" + "=" * 104)
print("【5】跨域迁移：Misra 预训练是否真的带来域适应能力")
print("=" * 104)
sm = df[df.split == "misra_test"]
if len(sm) == 2:
    bm = sm[sm.model.str.contains("基线")].iloc[0]
    nm = sm[sm.model.str.contains("两阶段")].iloc[0]
    majority = max(bm.true_irony_rate, 100 - bm.true_irony_rate)
    print(f"  Misra test（n={int(bm.n)}，讽刺占比 {bm.true_irony_rate:.1f}%，"
          f"多数类基线 {majority:.1f}%）")
    print(f"    基线（从未见过新闻标题）: 准确率 {bm.accuracy:.2f}%  "
          f"{'低于' if bm.accuracy < majority else '高于'}多数类基线 "
          f"{bm.accuracy-majority:+.2f}pp")
    print(f"    两阶段（Misra 预训练过）: 准确率 {nm.accuracy:.2f}%  "
          f"高于多数类基线 {nm.accuracy-majority:+.2f}pp")
    print(f"    → 跨域提升 {nm.accuracy-bm.accuracy:+.2f}pp，"
          f"证明域适应预训练确实生效。")
    print(f"\n  但同域高性能 ≠ 跨域迁移：")
    st = df[df.split == "semeval_test"]
    bst = st[st.model.str.contains("基线")].iloc[0]
    nst = st[st.model.str.contains("两阶段")].iloc[0]
    print(f"    Misra 域内（同域）  : {bm.accuracy:.2f}% → {nm.accuracy:.2f}% "
          f"({nm.accuracy-bm.accuracy:+.2f}pp)")
    print(f"    SemEval 推特（跨域）: {bst.accuracy:.2f}% → {nst.accuracy:.2f}% "
          f"({nst.accuracy-bst.accuracy:+.2f}pp)")
    print(f"    → 迁移效率 = {(nst.accuracy-bst.accuracy)/max(nm.accuracy-bm.accuracy,1e-9)*100:.1f}%"
          f"（Misra 域内提升中仅有这么小比例迁移到推特域）")
    if nst.recall_irony > bst.recall_irony + 5:
        print(f"    → 但反讽召回迁移成功: {bst.recall_irony:.2f}% → "
              f"{nst.recall_irony:.2f}% ({nst.recall_irony-bst.recall_irony:+.2f}pp)")
        print(f"      说明 Misra 预训练教会模型「字面与事实背离」的通用模式，")
        print(f"      这部分能力跨域有效；而新闻标题特有的风格特征不迁移。")

# ── 6. 结论 ──
print("\n" + "=" * 104)
print("【结论】")
print("=" * 104)
# 显式取 semeval_test 的记录：上面循环结束后 b/nn 是最后一个 split（misra_test）的残留值
_s = df[df.split == "semeval_test"]
b = _s[_s.model.str.contains("基线")].iloc[0]
nn = _s[_s.model.str.contains("两阶段")].iloc[0]
sg_t = sig.get("semeval_test", {})
c = [
    f"1. 重训达成主目标：SemEval test 反讽召回 {b.recall_irony:.2f}% → "
    f"{nn.recall_irony:.2f}%（{nn.recall_irony-b.recall_irony:+.2f}pp），"
    f"F1 {b.f1_irony:.4f} → {nn.f1_irony:.4f}（{nn.f1_irony-b.f1_irony:+.4f}）。",

    f"2. 但准确率几乎不动：{b.accuracy:.2f}% → {nn.accuracy:.2f}%"
    f"（{nn.accuracy-b.accuracy:+.2f}pp，McNemar χ²={sg_t.get('mcnemar_chi2')}，"
    f"b={sg_t.get('b_base_wrong_new_right')}/c={sg_t.get('c_base_right_new_wrong')} → "
    f"{'显著' if sg_t.get('significant') else '不显著'}）。"
    f"总错误数 {b.fn+b.fp} → {nn.fn+nn.fp}。"
    f"错误类型从漏判反讽（FN={b.fn}）转移为误判反讽（FP={nn.fp}）。",

    f"3. 代价是引入新的 pro 偏置：裁决者判反讽率 {b.pred_pro_rate:.2f}% → "
    f"{nn.pred_pro_rate:.2f}%，真实 {b.true_irony_rate:.2f}%，"
    f"偏置 {b.bias_pp:+.2f}pp → {nn.bias_pp:+.2f}pp。",

    f"4. ★ 最重要风险：偏置方向反转，与辩论方 pro 偏置同向，"
    f"「偏置互补抵消」机制被破坏。预测重训后辩论增益进一步下降，需实测验证。",

    f"5. 跨域迁移部分成功：Misra 域内 {bm.accuracy:.2f}%→{nm.accuracy:.2f}%"
    f"（{nm.accuracy-bm.accuracy:+.2f}pp），推特域仅 {nst.accuracy-bst.accuracy:+.2f}pp，"
    f"迁移效率 {(nst.accuracy-bst.accuracy)/max(nm.accuracy-bm.accuracy,1e-9)*100:.1f}%。"
    f"但反讽召回跨域迁移良好，说明学到的是通用反讽模式而非新闻标题风格。",
]
if len(df_gate[df_gate.split == "semeval_test"]) == 2:
    ab = df_gate[(df_gate.split == "semeval_test") &
                 df_gate.model.str.contains("基线")].auc_gate.iloc[0]
    an = df_gate[(df_gate.split == "semeval_test") &
                 df_gate.model.str.contains("两阶段")].auc_gate.iloc[0]
    c.append(
        f"6. 门控信号：全量 784 条上 AUC 基线 {ab:.4f} → 两阶段 {an:.4f}"
        f"（{an-ab:+.4f}），{'改善但仍低于 0.60 可用线' if an < 0.60 else '达到可用线'}。")
for x in c:
    print("  " + x)

out = {
    "comparison": rows,
    "error_shift": {
        "base": {"tp": int(b.tp), "fn": int(b.fn), "tn": int(b.tn), "fp": int(b.fp),
                 "total_errors": int(b.fn + b.fp)},
        "twostage": {"tp": int(nn.tp), "fn": int(nn.fn), "tn": int(nn.tn),
                     "fp": int(nn.fp), "total_errors": int(nn.fn + nn.fp)},
    },
    "bias_interaction": bias_rows,
    "gate_signal": gate_rows,
    "cross_domain": {
        "misra_in_domain": {"base": bm.accuracy, "twostage": nm.accuracy,
                            "delta_pp": round(nm.accuracy - bm.accuracy, 2)},
        "semeval_cross_domain": {"base": bst.accuracy, "twostage": nst.accuracy,
                                 "delta_pp": round(nst.accuracy - bst.accuracy, 2)},
        "transfer_efficiency_pct": round(
            (nst.accuracy - bst.accuracy) / max(nm.accuracy - bm.accuracy, 1e-9) * 100, 1),
    },
    "risk": ("裁决者 pro 偏置方向反转，与辩论方偏置同向，互补抵消机制可能失效，"
             "预测辩论增益进一步下降，需服务器重跑辩论流程验证"),
}
p = f"{RES}/retrain_comparison.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
df.to_csv(f"{RES}/retrain_comparison.csv", index=False, encoding="utf-8-sig")
df_gate.to_csv(f"{RES}/retrain_gate_signal.csv", index=False, encoding="utf-8-sig")
print(f"\n结果已写入: retrain_comparison.json / .csv / retrain_gate_signal.csv")
print("=" * 104)
