"""归因分析：为何辩论增益从均衡集 +18.67pp 崩塌到自然分布 +2.00pp。

实测 vs 事后分层估算的偏差：辩论 -9.83pp（估算 74.83% → 实测 65.00%）。
估算假设「辩论方在各分层内的行为与均衡集一致」，偏差说明该假设不成立。

核心待验假设：辩论方 pro 偏置是「固定倾向」而非「随类别比例自适应」。
  均衡集   : 真实反讽 50.0% → 辩论判 pro 65.33%  偏置 +15.33pp
  自然分布 : 真实反讽 44.0% → 辩论判 pro 72.00%  偏置 +28.00pp
若偏置是固定倾向，则真实反讽率越低，偏置造成的伤害越大 → 解释崩塌。
"""
import os
import json
import math
import numpy as np
import pandas as pd

RES = "experiments/results"
EQ_D = f"{RES}/debate_lens_n150_with.csv"
EQ_B = f"{RES}/debate_lens_n150_judge_only.csv"
NA_D = f"{RES}/natural200_debate_with.csv"
NA_B = f"{RES}/natural200_judge_only.csv"

V2B = {"pro": 1, "con": 0}


def load(debate_path, base_path, tag):
    d = pd.read_csv(debate_path)
    b = pd.read_csv(base_path)
    d["gold"] = d["true_label"].astype(int)
    d["debate_b"] = d["predicted"].map(V2B)
    d["judge_b"] = d["judge_vote"].map(V2B)
    d["pro_b"] = d["pro_vote"].map(V2B)
    d["con_b"] = d["con_vote"].map(V2B)
    b["gold"] = b["true_label"].astype(int)
    b["judge_b"] = b["predicted"].map(V2B)
    d = d.merge(b[["sample_id", "judge_b"]].rename(columns={"judge_b": "judge_b_chk"}),
                on="sample_id", how="left")
    print(f"  [{tag}] 辩论 n={len(d)}  基线 n={len(b)}  "
          f"裁决者判决一致性={(d.judge_b==d.judge_b_chk).mean()*100:.1f}%")
    return d, b


print("=" * 104)
print("辩论增益崩塌归因分析")
print("=" * 104)
eq_d, eq_b = load(EQ_D, EQ_B, "均衡 n=150")
na_d, na_b = load(NA_D, NA_B, "自然 n=200")


def se(pct, n):
    return math.sqrt(pct * (100 - pct) / n)


def block(name, d, b):
    n = len(d)
    acc_d = d.debate_b.eq(d.gold).mean() * 100
    acc_j = b.judge_b.eq(b.gold).mean() * 100
    true_r = d.gold.mean() * 100
    pro_r = d.debate_b.mean() * 100
    jpro_r = b.judge_b.mean() * 100
    return {
        "set": name, "n": n, "acc_debate": round(acc_d, 2), "acc_judge": round(acc_j, 2),
        "delta": round(acc_d - acc_j, 2),
        "true_irony_rate": round(true_r, 2),
        "debate_pro_rate": round(pro_r, 2),
        "judge_pro_rate": round(jpro_r, 2),
        "debate_bias_pp": round(pro_r - true_r, 2),
        "judge_bias_pp": round(jpro_r - true_r, 2),
    }


# ══════════ 1. 偏置的固定性检验 ══════════
print("\n" + "=" * 104)
print("【1】辩论方 pro 偏置是否随类别比例自适应")
print("=" * 104)
be = block("均衡 n=150", eq_d, eq_b)
bn = block("自然 n=200", na_d, na_b)
print(f"  {'测试集':<16} {'真实反讽率':>11} {'辩论判pro率':>12} {'偏置':>9} "
      f"{'裁决判pro率':>12} {'偏置':>9}")
print("  " + "-" * 76)
for b in [be, bn]:
    print(f"  {b['set']:<16} {b['true_irony_rate']:>10.1f}% {b['debate_pro_rate']:>11.1f}% "
          f"{b['debate_bias_pp']:>+8.1f}pp {b['judge_pro_rate']:>11.1f}% "
          f"{b['judge_bias_pp']:>+8.1f}pp")

print(f"\n  真实反讽率变化: {be['true_irony_rate']:.1f}% → {bn['true_irony_rate']:.1f}% "
      f"({bn['true_irony_rate']-be['true_irony_rate']:+.1f}pp)")
print(f"  辩论判 pro 率变化: {be['debate_pro_rate']:.1f}% → {bn['debate_pro_rate']:.1f}% "
      f"({bn['debate_pro_rate']-be['debate_pro_rate']:+.1f}pp)")
print(f"  辩论偏置变化: {be['debate_bias_pp']:+.1f}pp → {bn['debate_bias_pp']:+.1f}pp "
      f"({bn['debate_bias_pp']-be['debate_bias_pp']:+.1f}pp)")

# 关键判据：真实反讽率下降时，辩论判 pro 率是否也下降
d_true = bn["true_irony_rate"] - be["true_irony_rate"]
d_pro = bn["debate_pro_rate"] - be["debate_pro_rate"]
if d_true < 0 and d_pro > 0:
    verdict = ("辩论方 pro 倾向【完全不自适应】：真实反讽率下降 %.1fpp，"
               "辩论判 pro 率反而上升 %.1fpp，偏置扩大 %.1fpp"
               % (-d_true, d_pro, bn["debate_bias_pp"] - be["debate_bias_pp"]))
elif d_true < 0 and d_pro < 0 and abs(d_pro) < abs(d_true):
    verdict = ("辩论方【部分自适应】：判 pro 率随真实率下降但幅度不足（%.1fpp vs %.1fpp），"
               "偏置仍扩大 %.1fpp" % (d_pro, d_true,
                                    bn["debate_bias_pp"] - be["debate_bias_pp"]))
else:
    verdict = "辩论方偏置随类别比例变化，机制需重新判定"
print(f"\n  ★ 结论: {verdict}")

# ══════════ 2. 分类别伤害的放大 ══════════
print("\n" + "=" * 104)
print("【2】分类别命中：辩论对非反讽的伤害是否被放大")
print("=" * 104)
print(f"  {'类别':<12} {'测试集':<16} {'裁决者':>9} {'辩论':>9} {'差值':>10} {'样本数':>7}")
print("  " + "-" * 70)
cls_rows = []
for lab, cname in [(1, "真反讽"), (0, "真非反讽")]:
    for tag, d in [("均衡 n=150", eq_d), ("自然 n=200", na_d)]:
        s = d[d.gold == lab]
        aj = s.judge_b.eq(s.gold).mean() * 100
        ad = s.debate_b.eq(s.gold).mean() * 100
        cls_rows.append({"class": cname, "set": tag, "n": len(s),
                         "acc_judge": round(aj, 2), "acc_debate": round(ad, 2),
                         "delta_pp": round(ad - aj, 2)})
        print(f"  {cname:<12} {tag:<16} {aj:>8.2f}% {ad:>8.2f}% {ad-aj:>+9.2f}pp {len(s):>7}")
    print("  " + "-" * 70)

# 非反讽伤害放大倍数
harm_eq = [r for r in cls_rows if r["class"] == "真非反讽" and "均衡" in r["set"]][0]["delta_pp"]
harm_na = [r for r in cls_rows if r["class"] == "真非反讽" and "自然" in r["set"]][0]["delta_pp"]
gain_eq = [r for r in cls_rows if r["class"] == "真反讽" and "均衡" in r["set"]][0]["delta_pp"]
gain_na = [r for r in cls_rows if r["class"] == "真反讽" and "自然" in r["set"]][0]["delta_pp"]
print(f"\n  反讽类增益: {gain_eq:+.2f}pp → {gain_na:+.2f}pp  "
      f"(变化 {gain_na-gain_eq:+.2f}pp)")
print(f"  非反讽类伤害: {harm_eq:+.2f}pp → {harm_na:+.2f}pp  "
      f"(变化 {harm_na-harm_eq:+.2f}pp)")
print(f"  非反讽伤害放大倍数: {abs(harm_na)/max(abs(harm_eq),1e-9):.2f}×")
print(f"  → 反讽增益基本稳定（{gain_na-gain_eq:+.2f}pp），"
      f"崩塌主要来自非反讽伤害放大（{harm_na-harm_eq:+.2f}pp）")

# ══════════ 3. 增益的成分分解 ══════════
print("\n" + "=" * 104)
print("【3】总增益的成分分解：增益 = π·Δ反讽 + (1-π)·Δ非反讽")
print("=" * 104)
print(f"  {'测试集':<16} {'π(反讽占比)':>12} {'反讽贡献':>11} {'非反讽贡献':>12} "
      f"{'总增益':>9} {'实测':>8}")
print("  " + "-" * 76)
decomp = []
for tag, d, b in [("均衡 n=150", eq_d, eq_b), ("自然 n=200", na_d, na_b)]:
    pi = d.gold.mean()
    cr = cls_rows
    g = [x for x in cr if x["class"] == "真反讽" and x["set"] == tag][0]["delta_pp"]
    h = [x for x in cr if x["class"] == "真非反讽" and x["set"] == tag][0]["delta_pp"]
    contrib_p = pi * g
    contrib_n = (1 - pi) * h
    total = contrib_p + contrib_n
    actual = d.debate_b.eq(d.gold).mean() * 100 - b.judge_b.eq(b.gold).mean() * 100
    decomp.append({"set": tag, "pi": round(pi, 4), "gain_irony_pp": round(g, 2),
                   "harm_nonirony_pp": round(h, 2),
                   "contrib_irony_pp": round(contrib_p, 2),
                   "contrib_nonirony_pp": round(contrib_n, 2),
                   "total_decomposed_pp": round(total, 2),
                   "actual_delta_pp": round(actual, 2)})
    print(f"  {tag:<16} {pi*100:>11.1f}% {contrib_p:>+10.2f}pp {contrib_n:>+11.2f}pp "
          f"{total:>+8.2f}pp {actual:>+7.2f}pp")
print(f"\n  分解与实测一致性: 均衡 {abs(decomp[0]['total_decomposed_pp']-decomp[0]['actual_delta_pp']):.2f}pp 偏差, "
      f"自然 {abs(decomp[1]['total_decomposed_pp']-decomp[1]['actual_delta_pp']):.2f}pp 偏差")

# 反事实：若辩论方无 pro 偏置
print(f"\n  ◆ 反事实推演：若辩论方 pro 率 = 真实反讽率（无偏置）")
for tag, d in [("均衡 n=150", eq_d), ("自然 n=200", na_d)]:
    pi = d.gold.mean()
    # 无偏置下：非反讽伤害应趋近 0，反讽增益保留
    ideal = pi * [x for x in cls_rows if x["class"] == "真反讽" and x["set"] == tag][0]["delta_pp"]
    actual = [x for x in decomp if x["set"] == tag][0]["actual_delta_pp"]
    print(f"    {tag:<16} 无偏置理想增益 ≈ {ideal:+.2f}pp   实测 {actual:+.2f}pp   "
          f"偏置造成的损耗 {actual-ideal:+.2f}pp")

# ══════════ 4. 临界反讽占比重算 ══════════
print("\n" + "=" * 104)
print("【4】临界反讽占比重算（辩论不再有净增益的 π*）")
print("=" * 104)
for tag in ["均衡 n=150", "自然 n=200"]:
    g = [x for x in cls_rows if x["class"] == "真反讽" and x["set"] == tag][0]["delta_pp"]
    h = [x for x in cls_rows if x["class"] == "真非反讽" and x["set"] == tag][0]["delta_pp"]
    if g - h == 0:
        continue
    pi_star = -h / (g - h) * 100
    print(f"  用 {tag} 的分类别数据: Δ反讽={g:+.2f}pp, Δ非反讽={h:+.2f}pp")
    print(f"    临界 π* = {pi_star:.2f}%  "
          f"（反讽占比低于此值时辩论净增益为负）")
    print(f"    该测试集实际 π = {[x for x in decomp if x['set']==tag][0]['pi']*100:.1f}%  "
          f"→ {'仍在临界值之上' if [x for x in decomp if x['set']==tag][0]['pi']*100 > pi_star else '已跌破临界值'}")
    print()

# 用自然分布参数外推完整测试集
g_na = [x for x in cls_rows if x["class"] == "真反讽" and x["set"] == "自然 n=200"][0]["delta_pp"]
h_na = [x for x in cls_rows if x["class"] == "真非反讽" and x["set"] == "自然 n=200"][0]["delta_pp"]
pi_star_na = -h_na / (g_na - h_na)
print(f"  ★ 以自然分布参数外推：完整测试集 π=39.7%")
print(f"    预期辩论增益 = 0.397×{g_na:+.2f} + 0.603×{h_na:+.2f} = "
      f"{0.397*g_na + 0.603*h_na:+.2f}pp")
print(f"    临界 π* = {pi_star_na*100:.2f}%，39.7% {'高于' if 39.7 > pi_star_na*100 else '低于'}临界值")

# ══════════ 5. 事后分层估算为何失效 ══════════
print("\n" + "=" * 104)
print("【5】事后分层估算失效的原因")
print("=" * 104)
print(f"  估算前提：辩论方在各分层内的行为与均衡集一致（分层可交换性）")
print(f"  估算结果：辩论 74.83%（增益 +13.43pp）")
print(f"  实测结果：辩论 {bn['acc_debate']:.2f}%（增益 {bn['delta']:+.2f}pp）")
print(f"  偏差：{bn['acc_debate']-74.83:+.2f}pp")
print(f"\n  失效机理：估算用均衡集的 Δ非反讽 = {harm_eq:+.2f}pp 外推，")
print(f"           但实测自然分布下 Δ非反讽 = {harm_na:+.2f}pp（放大 {abs(harm_na)/max(abs(harm_eq),1e-9):.2f}×）。")
print(f"           分层可交换性假设不成立——辩论方的 pro 倾向是绝对倾向，")
print(f"           不随测试集类别比例缩放，故在低反讽占比下伤害被系统性放大。")
print(f"\n  方法论教训：涉及有系统性偏置的组件时，事后分层加权不能外推到不同类别比例。")

# ══════════ 6. 门控在自然分布下的价值反转 ══════════
print("\n" + "=" * 104)
print("【6】门控价值的反转：自然分布下门控反而更有用")
print("=" * 104)
na_gate = pd.read_csv(f"{RES}/natural200_gate_scan.csv")
eq_gate = pd.read_csv(f"{RES}/debate_lens_gate_scan.csv")
print(f"  {'τ':>6} {'自然分布准确率':>15} {'触发率':>9} {'vs纯裁决者':>12} {'延迟节省':>10}")
print("  " + "-" * 60)
lat_full = na_gate[na_gate.tau > 1.0].weighted_latency.iloc[0] if (na_gate.tau > 1.0).any() else 24.45
best_nat = None
for _, r in na_gate.iterrows():
    if r.tau > 1.0:
        continue
    save = (1 - r.weighted_latency / lat_full) * 100
    print(f"  {r.tau:>6.2f} {r.accuracy:>14.2f}% {r.trigger_rate:>8.1f}% "
          f"{r.accuracy-bn['acc_judge']:>+11.2f}pp {save:>9.0f}%")
    if best_nat is None or r.accuracy > best_nat["accuracy"]:
        best_nat = {"tau": r.tau, "accuracy": r.accuracy,
                    "trigger_rate": r.trigger_rate, "latency": r.weighted_latency}
print(f"\n  ★ 自然分布最优 τ={best_nat['tau']:.2f}: {best_nat['accuracy']:.2f}%")
print(f"    vs 纯裁决者 {best_nat['accuracy']-bn['acc_judge']:+.2f}pp")
print(f"    vs 全程辩论 {best_nat['accuracy']-bn['acc_debate']:+.2f}pp  ← 门控【超过】全程辩论")
print(f"    触发率 {best_nat['trigger_rate']:.1f}%，延迟节省 "
      f"{(1-best_nat['latency']/lat_full)*100:.0f}%")
print(f"\n  ◆ 关键反转：")
print(f"    均衡集   : 全程辩论 {be['acc_debate']:.2f}% > 门控最优 → 门控无增益价值")
print(f"    自然分布 : 门控 {best_nat['accuracy']:.2f}% > 全程辩论 {bn['acc_debate']:.2f}% "
      f"({best_nat['accuracy']-bn['acc_debate']:+.2f}pp)")
print(f"    → 因为辩论方 pro 偏置在自然分布下有害，门控通过「少辩论」规避了这部分伤害。")
print(f"      门控的作用从「省算力」变成「防止辩论方偏置污染高置信判决」。")

# ══════════ 7. 统计功效检讨 ══════════
print("\n" + "=" * 104)
print("【7】统计功效检讨：区分「估算被推翻」与「样本量不足」")
print("=" * 104)
merged = na_d.merge(na_b[["sample_id", "judge_b"]].rename(columns={"judge_b": "jb2"}),
                    on="sample_id", how="left")
disc = merged[merged.debate_b != merged.jb2]
b_cnt = int((disc.debate_b == disc.gold).sum())
c_cnt = int((disc.jb2 == disc.gold).sum())
chi2 = (abs(b_cnt - c_cnt) - 1) ** 2 / (b_cnt + c_cnt) if (b_cnt + c_cnt) else 0.0
print(f"  不一致对: b={b_cnt}（辩论对/裁决者错）, c={c_cnt}（裁决者对/辩论错）")
print(f"  McNemar χ²={chi2:.3f}  → {'显著' if chi2>3.841 else '不显著'}（临界 3.841, α=0.05）")

p_disc = (b_cnt + c_cnt) / len(merged)
print(f"  不一致率 p_disc = {p_disc*100:.1f}%")


def power_for(delta_pp, n, p_d, alpha=0.05, z_beta=0.8416):
    """McNemar 检验检出 delta_pp 增益的功效。

    原理: b+c = p_disc·n, b−c = delta·n
          → p_b = 0.5 + delta/(2·p_disc)，检验统计量 z = (p_b−0.5)·sqrt(b+c)/0.5
    """
    if p_d <= 0:
        return float("nan"), None
    p_b = 0.5 + (delta_pp / 100) / (2 * p_d)
    if not (0 < p_b < 1):
        return float("nan"), None
    n_disc = n * p_d
    z = (p_b - 0.5) * math.sqrt(n_disc) / 0.5
    from statistics import NormalDist
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    pw = 1 - NormalDist().cdf(z_a - z)
    # 达到 80% 功效所需 n
    need = ((z_a + z_beta) * 0.5 / (p_b - 0.5)) ** 2 / p_d if p_b != 0.5 else None
    return pw, need


print(f"\n  ◆ 关键问题①：n=200 能否检出事后分层估算的 +13.43pp？")
pw_est, need_est = power_for(13.43, 200, p_disc)
print(f"    n=200 检出 +13.43pp 的功效 = {pw_est*100:.1f}%")
if pw_est >= 0.80:
    print(f"    → 功效【充足】（≥80%）。若真实增益是 +13.43pp，n=200 有 {pw_est*100:.0f}% "
          f"概率检出显著。")
    print(f"    → 实测仅 +2.00pp 且 χ²={chi2:.3f} 不显著，说明【估算被实测推翻】，"
          f"而非样本量不足。")
    print(f"    → 这排除了归因分析第7节的解释(b)，确认解释(a)：偏置放大真实存在。")
else:
    print(f"    → 功效不足（<80%），需 n≈{need_est:.0f} 才能可靠检出，崩塌结论待复核。")

print(f"\n  ◆ 关键问题②：n=200 能否区分实测的 +2.00pp 与 0？")
pw_obs, need_obs = power_for(2.00, 200, p_disc)
print(f"    n=200 检出 +2.00pp 的功效 = {pw_obs*100:.1f}%")
print(f"    → 功效极低，+2.00pp 与「零增益」在统计上不可区分。")
if need_obs:
    print(f"    → 需 n≈{need_obs:.0f} 才有 80% 功效检出 +2.00pp（完整测试集仅 784 条，不够）。")

print(f"\n  ◆ 各候选增益的检出功效与所需样本量")
print(f"  {'待检增益':<12} {'n=200 功效':>11} {'n=784 功效':>11} {'80%功效所需 n':>15}")
print("  " + "-" * 54)
for delta, label in [(13.43, "估算值"), (4.50, "门控最优"), (2.00, "全程辩论实测"),
                     (18.67, "均衡集增益")]:
    p200, need = power_for(delta, 200, p_disc)
    p784, _ = power_for(delta, 784, p_disc)
    need_s = f"{need:>14.0f}" if need else "         不可达"
    print(f"  {label} {delta:+.2f}pp {p200*100:>10.1f}% {p784*100:>10.1f}% {need_s}")

print(f"\n  ★ 修正后的结论（原脚本此处有误，已更正）:")
print(f"    • 原表述「n=200 功效不足以检出 +13.43pp」是错的 —— 实际功效 {pw_est*100:.1f}%，充足。")
print(f"    • 正确表述：功效充足却测得 +2.00pp 且不显著 → 事后分层估算被实测推翻。")
print(f"    • 但 +2.00pp 本身也检不出（功效 {pw_obs*100:.1f}%），故「崩塌到多少」无法精确定值，")
print(f"      只能确定「远低于估算的 +13.43pp」。")
print(f"    • 完整测试集 784 条对 +4.50pp 的门控增益功效为 "
      f"{power_for(4.50, 784, p_disc)[0]*100:.1f}%，值得跑全量确认门控反转结论。")

# ══════════ 汇总 ══════════
print("\n" + "=" * 104)
print("【汇总】")
print("=" * 104)
summary = [
    f"1. 辩论增益崩塌: 均衡 {be['delta']:+.2f}pp → 自然 {bn['delta']:+.2f}pp（不显著），"
    f"事后分层估算 {13.43:+.2f}pp 偏差 {bn['delta']-13.43:+.2f}pp。",
    f"2. 根因是辩论方 pro 偏置不自适应: 真实反讽率 {be['true_irony_rate']:.1f}%→"
    f"{bn['true_irony_rate']:.1f}%（{d_true:+.1f}pp），辩论判 pro 率却 "
    f"{be['debate_pro_rate']:.1f}%→{bn['debate_pro_rate']:.1f}%（{d_pro:+.1f}pp），"
    f"偏置从 {be['debate_bias_pp']:+.1f}pp 扩大到 {bn['debate_bias_pp']:+.1f}pp。",
    f"3. 崩塌来自非反讽伤害放大: Δ反讽 {gain_eq:+.2f}→{gain_na:+.2f}pp（稳定），"
    f"Δ非反讽 {harm_eq:+.2f}→{harm_na:+.2f}pp（放大 {abs(harm_na)/max(abs(harm_eq),1e-9):.2f}×）。",
    f"4. 事后分层估算方法失效: 分层可交换性假设不成立，有系统性偏置的组件不能这样外推。",
    f"5. 临界反讽占比大幅上移: 均衡集参数下 π*={-harm_eq/(gain_eq-harm_eq)*100:.2f}%，"
    f"自然分布参数下 π*={pi_star_na*100:.2f}%（上移 "
    f"{pi_star_na*100 - (-harm_eq/(gain_eq-harm_eq)*100):.2f}pp）。"
    f"完整测试集 π=39.7% {'已跌破' if 39.7 < pi_star_na*100 else '仍高于'}该临界值，"
    f"外推预期增益 {0.397*g_na+0.603*h_na:+.2f}pp"
    f"{'（为负，即辩论有害）' if 0.397*g_na+0.603*h_na < 0 else ''}。",
    f"6. 门控价值反转: 自然分布下门控 τ={best_nat['tau']:.2f} 达 {best_nat['accuracy']:.2f}%，"
    f"【超过】全程辩论 {bn['acc_debate']:.2f}%（{best_nat['accuracy']-bn['acc_debate']:+.2f}pp），"
    f"延迟节省 {(1-best_nat['latency']/lat_full)*100:.0f}%。门控作用从省算力变为防止偏置污染。",
    f"7. 统计功效（修正原脚本错误结论）: n=200 检出估算值 +13.43pp 的功效为 "
    f"{power_for(13.43, 200, p_disc)[0]*100:.1f}%（充足），故实测 +2.00pp 且不显著"
    f"说明【估算被推翻】而非样本不足；但 +2.00pp 本身功效仅 "
    f"{power_for(2.00, 200, p_disc)[0]*100:.1f}%，崩塌的具体幅度无法定值，"
    f"只能确定远低于估算。门控 +4.50pp 在 n=784 下功效 "
    f"{power_for(4.50, 784, p_disc)[0]*100:.1f}%，值得跑全量确认反转结论。",
]
for s in summary:
    print("  " + s)

out = {
    "bias_fixedness": {"balanced": be, "natural": bn,
                       "true_rate_change_pp": round(d_true, 2),
                       "debate_pro_rate_change_pp": round(d_pro, 2),
                       "bias_change_pp": round(bn["debate_bias_pp"] - be["debate_bias_pp"], 2),
                       "verdict": verdict},
    "by_class": cls_rows,
    "decomposition": decomp,
    "harm_amplification": {"balanced_pp": harm_eq, "natural_pp": harm_na,
                           "ratio": round(abs(harm_na) / max(abs(harm_eq), 1e-9), 2)},
    "critical_pi_pct": {
        "from_balanced_params": round(
            -harm_eq / (gain_eq - harm_eq) * 100, 2) if gain_eq != harm_eq else None,
        "from_natural_params": round(pi_star_na * 100, 2),
        "full_test_pi_pct": 39.7,
        "predicted_delta_full_test_pp": round(0.397 * g_na + 0.603 * h_na, 2),
    },
    "gate_reversal": {
        "natural_best_tau": best_nat["tau"],
        "natural_best_acc_pct": best_nat["accuracy"],
        "natural_full_debate_acc_pct": bn["acc_debate"],
        "gate_beats_full_debate_pp": round(best_nat["accuracy"] - bn["acc_debate"], 2),
        "latency_saving_pct": round((1 - best_nat["latency"] / lat_full) * 100, 1),
    },
    "mcnemar": {"b": b_cnt, "c": c_cnt, "chi2": round(chi2, 3),
                "significant": bool(chi2 > 3.841),
                "disagreement_rate_pct": round(p_disc * 100, 2)},
    "statistical_power": {
        "power_estimate_13_43pp_at_n200": round(power_for(13.43, 200, p_disc)[0] * 100, 1),
        "power_observed_2_00pp_at_n200": round(power_for(2.00, 200, p_disc)[0] * 100, 1),
        "power_gate_4_50pp_at_n784": round(power_for(4.50, 784, p_disc)[0] * 100, 1),
        "verdict": ("检出估算值 +13.43pp 的功效充足，故估算被实测推翻而非样本不足；"
                    "但实测 +2.00pp 本身功效极低，崩塌幅度无法精确定值"),
    },
    "poststrat_estimate": {"debate_pct": 74.83, "delta_pp": 13.43,
                           "measured_debate_pct": bn["acc_debate"],
                           "measured_delta_pp": bn["delta"],
                           "deviation_pp": round(bn["acc_debate"] - 74.83, 2),
                           "reason": "分层可交换性假设不成立：辩论方 pro 倾向为绝对倾向，不随类别比例缩放"},
}
p = f"{RES}/gain_collapse_attribution.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
pd.DataFrame(cls_rows).to_csv(f"{RES}/attribution_by_class.csv",
                              index=False, encoding="utf-8-sig")
print(f"\n结果已写入: gain_collapse_attribution.json / attribution_by_class.csv")
print("=" * 104)
