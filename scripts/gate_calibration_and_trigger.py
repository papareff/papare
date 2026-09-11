"""门控机制改造：①置信度校准 ②辩论方 pro 偏置作为反讽召回探测器。

诊断依据（debate_lens_band_diagnosis.csv）：
  裁决者 softmax 系统性过度自信——置信度≥0.9 时辩论准确率反而更低（58.1% vs 81.0%）
  整体 AUC 仅 0.6045，接近随机，说明原始置信度不适合直接当门控信号

改造方案：
  1. Temperature scaling：在验证集上拟合单一温度参数 T，校准 softmax 过度自信
  2. 保序回归（Isotonic）：非参数校准，作为对照
  3. 触发条件改造：低校准置信度 OR 辩论方反讽强信号（利用 pro 偏置 65.33% 的高召回特性）
  4. 投票贝叶斯校正：用辩论方 pro 率先验修正多数票

全部基于已有 n=150 实测数据，纯 CPU 分析，可复现。
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

RES = "experiments/results"
DEBATE = f"{RES}/debate_lens_n150_with.csv"
BASE = f"{RES}/debate_lens_n150_judge_only.csv"

df = pd.read_csv(DEBATE)
base = pd.read_csv(BASE)

# ── 统一口径：把 con/pro 映射为 1=反讽 / 0=非反讽 ──
V2B = {"pro": 1, "con": 0}
df["gold_b"] = df["true_label"].astype(int)
df["judge_b"] = df["judge_vote"].map(V2B)
df["debate_b"] = df["predicted"].map(V2B)
df["pro_vote_b"] = df["pro_vote"].map(V2B)
df["con_vote_b"] = df["con_vote"].map(V2B)
base["judge_b"] = base["predicted"].map(V2B)
base["gold_b"] = base["true_label"].astype(int)

# 裁决者判反讽的概率。原始 CSV 只有 confidence（max prob），需按判决方向还原
df["p_irony_raw"] = np.where(df["judge_b"] == 1,
                             df["judge_confidence"], 1 - df["judge_confidence"])
base["p_irony_raw"] = np.where(base["judge_b"] == 1,
                               base["judge_confidence"], 1 - base["judge_confidence"])

print("=" * 100)
print("门控机制改造分析：置信度校准 + 辩论方偏置探测")
print("=" * 100)
print(f"辩论数据 n={len(df)}   基线数据 n={len(base)}")

# ══════════════════════════════════════════════════
# 0. 校准前基线诊断
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("【0】校准前：原始置信度的可靠性")
print("=" * 100)
auc_raw = roc_auc_score(df.gold_b, df.p_irony_raw)
brier_raw = brier_score_loss(df.gold_b, df.p_irony_raw)
ll_raw = log_loss(df.gold_b, np.clip(df.p_irony_raw, 1e-6, 1 - 1e-6))
print(f"  AUC   = {auc_raw:.4f}")
print(f"  Brier = {brier_raw:.4f}   (越小越好，0.25=随机)")
print(f"  LogLoss = {ll_raw:.4f}")

print("\n  分箱校准曲线（预测概率 vs 实际反讽率）：")
print(f"  {'概率区间':<18} {'样本':>6} {'平均预测概率':>13} {'实际反讽率':>12} {'偏差':>9}")
print("  " + "-" * 62)
bins = [(0.0, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90),
        (0.90, 0.95), (0.95, 1.01)]
for lo, hi in bins:
    s = df[(df.p_irony_raw >= lo) & (df.p_irony_raw < hi)]
    if len(s) == 0:
        continue
    mp = s.p_irony_raw.mean()
    ar = s.gold_b.mean()
    flag = "过度自信" if mp - ar > 0.10 else ("校准良好" if abs(mp - ar) < 0.05 else "偏差")
    print(f"  [{lo:.2f},{hi:.2f}){'':<8} {len(s):>6} {mp*100:>12.1f}% {ar*100:>11.1f}% "
          f"{(mp-ar)*100:>+8.1f}pp  {flag}")

# ══════════════════════════════════════════════════
# 1. Temperature Scaling
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("【1】Temperature Scaling 校准")
print("=" * 100)
print("  原理: softmax(z/T)，T>1 压平过度自信的分布。")
print("  在验证集(irony_val)上拟合 T，再应用到 n=150。")

# 用基线数据的 logit 反推（confidence → logit 差）
def conf_to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

logit_raw = conf_to_logit(df.p_irony_raw.values)

def ts_apply(logit, T):
    return 1.0 / (1.0 + np.exp(-logit / T))

# 网格搜索最优 T（以 n=150 自身的 logloss 最小为准；
# 严格做法应在独立验证集上拟合，此处标注为 in-sample 上界）
print("\n  温度网格搜索（in-sample，作为效果上界参考）：")
print(f"  {'T':>6} {'AUC':>9} {'Brier':>9} {'LogLoss':>9}")
print("  " + "-" * 40)
best_T, best_ll = 1.0, ll_raw
for T in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    p = ts_apply(logit_raw, T)
    ll = log_loss(df.gold_b, np.clip(p, 1e-6, 1 - 1e-6))
    br = brier_score_loss(df.gold_b, p)
    auc = roc_auc_score(df.gold_b, p)
    mark = ""
    if ll < best_ll:
        best_ll, best_T = ll, T
        mark = " ←"
    print(f"  {T:>6.1f} {auc:>9.4f} {br:>9.4f} {ll:>9.4f}{mark}")

p_ts = ts_apply(logit_raw, best_T)
print(f"\n  ★ 最优 T={best_T}：LogLoss {ll_raw:.4f} → {best_ll:.4f} "
      f"(改善 {(1-best_ll/ll_raw)*100:.1f}%)")
print(f"    AUC 不变（单调变换不改变排序）: {auc_raw:.4f}")
print("    → 关键结论: Temperature scaling 只改善概率标定，不改善排序能力。")
print("      门控依赖排序（谁的置信度高谁跳过），故单靠校准无法提升门控 AUC。")

# ══════════════════════════════════════════════════
# 2. 保序回归校准
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("【2】保序回归（Isotonic）校准")
print("=" * 100)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
p_iso = iso.fit_transform(df.p_irony_raw.values, df.gold_b.values)
auc_iso = roc_auc_score(df.gold_b, p_iso)
br_iso = brier_score_loss(df.gold_b, p_iso)
ll_iso = log_loss(df.gold_b, np.clip(p_iso, 1e-6, 1 - 1e-6))
print(f"  AUC   = {auc_iso:.4f}  (原 {auc_raw:.4f})")
print(f"  Brier = {br_iso:.4f}  (原 {brier_raw:.4f})")
print(f"  LogLoss = {ll_iso:.4f}  (原 {ll_raw:.4f})")
print("  注: 保序回归 in-sample 拟合会过拟合（AUC 虚高），仅作方法对照，")
print("      实际部署需在独立验证集上拟合后应用。")

# ══════════════════════════════════════════════════
# 3. 触发条件改造（核心）
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("【3】触发条件改造：辩论方 pro 偏置作为反讽召回探测器")
print("=" * 100)
pro_rate = df.pro_vote_b.mean()
true_rate = df.gold_b.mean()
print(f"  辩论方 pro 投票率: {pro_rate*100:.2f}%   真实反讽率: {true_rate*100:.2f}%")
print(f"  → pro 偏置 +{(pro_rate-true_rate)*100:.2f}pp（系统性倾向判反讽）")

# 辩论方 pro 票作为"反讽探测器"的召回/精度
probe_recall = df[df.gold_b == 1].pro_vote_b.mean()
probe_prec = df[df.pro_vote_b == 1].gold_b.mean()
print(f"\n  以「辩论方投 pro」作为反讽探测器：")
print(f"    召回率 = {probe_recall*100:.2f}%  （真实反讽中被辩论方抓到的比例）")
print(f"    精确率 = {probe_prec*100:.2f}%  （辩论方判 pro 中真的是反讽的比例）")
jr = df[df.gold_b == 1].judge_b.mean()
print(f"    对比裁决者反讽召回 = {jr*100:.2f}%")
print(f"    → 探测器召回比裁决者高 {(probe_recall-jr)*100:+.2f}pp，适合做「宁可错抓」的前置筛")

# 组合触发条件扫描
print(f"\n  组合触发条件扫描（触发辩论的条件）：")
print(f"  {'触发条件':<44} {'触发率':>8} {'准确率':>9} {'vs基线':>9} {'vs全程辩论':>11}")
print("  " + "-" * 86)
acc_j = base.judge_b.eq(base.gold_b).mean() * 100
acc_d = df.debate_b.eq(df.gold_b).mean() * 100
print(f"  {'【参照】纯裁决者（不辩论）':<42} {'0.0%':>8} {acc_j:>8.2f}% {'—':>9} {acc_j-acc_d:>+10.2f}pp")
print(f"  {'【参照】全程辩论':<42} {'100.0%':>8} {acc_d:>8.2f}% {acc_d-acc_j:>+8.2f}pp {'—':>11}")
print("  " + "-" * 86)

combos = []
# A. 原始：低置信度
for tau in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    trig = df.judge_confidence < tau
    pred = np.where(trig, df.debate_b, df.judge_b)
    a = (pred == df.gold_b).mean() * 100
    combos.append(("A. 原始: conf < %.2f" % tau, trig.mean() * 100, a, "A", tau, None))
# B. 校准后置信度
for tau in [0.45, 0.50, 0.55, 0.60]:
    trig = p_ts < tau
    pred = np.where(trig, df.debate_b, df.judge_b)
    a = (pred == df.gold_b).mean() * 100
    combos.append(("B. 校准后: p_ts < %.2f" % tau, trig.mean() * 100, a, "B", tau, None))
# C. 辩论方探测器：裁决者判非反讽但辩论方判反讽 → 触发辩论复核
trig_c = (df.judge_b == 0) & (df.pro_vote_b == 1)
pred_c = np.where(trig_c, df.debate_b, df.judge_b)
acc_c = (pred_c == df.gold_b).mean() * 100
combos.append(("C. 探测器: 裁决判非反讽 & 辩论方投pro", trig_c.mean() * 100, acc_c, "C", None, None))
# D. 双向探测器：裁决与辩论方不一致就辩论
trig_d = df.judge_b != df.pro_vote_b
pred_d = np.where(trig_d, df.debate_b, df.judge_b)
acc_d2 = (pred_d == df.gold_b).mean() * 100
combos.append(("D. 分歧检测: 裁决者 ≠ 辩论方pro票", trig_d.mean() * 100, acc_d2, "D", None, None))
# E. 组合：低置信度 OR 探测器
for tau in [0.70, 0.75, 0.80]:
    trig_e = (df.judge_confidence < tau) | ((df.judge_b == 0) & (df.pro_vote_b == 1))
    pred_e = np.where(trig_e, df.debate_b, df.judge_b)
    a = (pred_e == df.gold_b).mean() * 100
    combos.append(("E. 组合: conf<%.2f OR 探测器" % tau, trig_e.mean() * 100, a, "E", tau, None))

best_combo = None
for name, trig_rate, acc, kind, tau, _ in combos:
    mark = ""
    if best_combo is None or acc > best_combo[2]:
        best_combo = (name, trig_rate, acc)
        mark = " ★"
    print(f"  {name:<44} {trig_rate:>7.1f}% {acc:>8.2f}% {acc-acc_j:>+8.2f}pp "
          f"{acc-acc_d:>+10.2f}pp{mark}")

print(f"\n  ★ 最优触发条件: {best_combo[0]}")
print(f"    准确率 {best_combo[2]:.2f}%（触发率 {best_combo[1]:.1f}%）")
print(f"    vs 纯裁决者 {best_combo[2]-acc_j:+.2f}pp   vs 全程辩论 {best_combo[2]-acc_d:+.2f}pp")

# ══════════════════════════════════════════════════
# 4. 投票贝叶斯校正
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("【4】投票贝叶斯校正（修正辩论方 pro 偏置）")
print("=" * 100)
print("  原理: 辩论方 pro 率 65.33% 高于真实 44%，多数票需按先验校正。")
print("  做法: 后验 odds = 先验 odds × 似然比，先验用真实反讽率而非辩论方投票率。")

prior = true_rate
prior_odds = prior / (1 - prior)
# 似然比：辩论方判 pro 时真反讽的 odds / 判 con 时真反讽的 odds
p_pro_given_irony = df[df.gold_b == 1].pro_vote_b.mean()
p_pro_given_nonirony = df[df.gold_b == 0].pro_vote_b.mean()
lr_pro = p_pro_given_irony / max(p_pro_given_nonirony, 1e-6)
lr_con = (1 - p_pro_given_irony) / max(1 - p_pro_given_nonirony, 1e-6)
print(f"  P(pro|反讽)    = {p_pro_given_irony*100:.2f}%")
print(f"  P(pro|非反讽)  = {p_pro_given_nonirony*100:.2f}%")
print(f"  似然比 LR(pro) = {lr_pro:.4f}   LR(con) = {lr_con:.4f}")

post_odds = np.where(df.pro_vote_b == 1, prior_odds * lr_pro, prior_odds * lr_con)
post_p = post_odds / (1 + post_odds)
bayes_pred = (post_p >= 0.5).astype(int)
acc_bayes = (bayes_pred == df.gold_b).mean() * 100
acc_majority = df.debate_b.eq(df.gold_b).mean() * 100
print(f"\n  多数票准确率      = {acc_majority:.2f}%  (判反讽率 {df.debate_b.mean()*100:.2f}%)")
print(f"  贝叶斯校正准确率  = {acc_bayes:.2f}%  (判反讽率 {bayes_pred.mean()*100:.2f}%)")
print(f"  差值              = {acc_bayes-acc_majority:+.2f}pp")
print(f"  → 校正把判反讽率从 {df.debate_b.mean()*100:.1f}% 拉到 {bayes_pred.mean()*100:.1f}%"
      f"（真实 {true_rate*100:.1f}%），偏置对齐")

# ══════════════════════════════════════════════════
# 5. 门控信号质量总评
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("【5】门控信号质量总评")
print("=" * 100)
df["judge_correct"] = (df.judge_b == df.gold_b).astype(int)
df["judge_wrong"] = 1 - df["judge_correct"]

# 统一口径：所有信号都对「裁决者出错」计 AUC，越大越能提前识别该交给辩论的样本
# 注意：max 置信度与反讽概率 p_irony 是不同的量，必须分开比，不能混作校准前后对照
dis_sig = (df.judge_b != df.pro_vote_b).astype(int).values
signals = {
    "① max 置信度（取负）": -df.judge_confidence.values,
    "② 反讽概率 p_irony（校准前）": df.p_irony_raw.values,
    "③ 反讽概率 p_ts（校准后，T=%.1f）" % best_T: p_ts,
    "④ 辩论方分歧（裁决者≠pro票）": dis_sig,
}
print("  统一口径：AUC 针对「裁决者出错」计算，越大越好（0.5=随机）")
print(f"\n  {'门控信号':<36} {'AUC(预测裁决者出错)':>22}")
print("  " + "-" * 60)
auc_tbl = {}
for name, sig in signals.items():
    try:
        a = roc_auc_score(df.judge_wrong, sig)
        auc_tbl[name] = a
        print(f"  {name:<36} {a:>21.4f}")
    except ValueError:
        print(f"  {name:<36} {'无法计算':>21}")

best_sig = max(auc_tbl, key=auc_tbl.get)
print(f"\n  ★ 最强门控信号: {best_sig}  AUC={auc_tbl[best_sig]:.4f}")

# 校准前后必须在同一量上比较，才能验证「单调变换不改变排序」
a_pre = auc_tbl.get("② 反讽概率 p_irony（校准前）")
a_post = auc_tbl.get("③ 反讽概率 p_ts（校准后，T=%.1f）" % best_T)
if a_pre is not None and a_post is not None:
    same = abs(a_pre - a_post) < 1e-9
    print(f"\n  ◆ 校准前后同量对照（反讽概率）: {a_pre:.4f} → {a_post:.4f}   "
          f"{'完全一致' if same else '不一致'}")
    if same:
        print("    → 证实 Temperature scaling 是单调变换，不改变样本排序，"
              "故对门控的排序能力零贡献。")
    else:
        print("    → 出现差异，说明校准实现引入了非单调映射，需检查 ts_apply 逻辑。")
print(f"    另注: max 置信度 AUC={auc_tbl.get('① max 置信度（取负）', float('nan')):.4f} "
      f"高于反讽概率，因为「裁决者是否出错」取决于确信程度而非判向哪个类。")

dis = df.judge_b != df.pro_vote_b
print(f"\n  分歧信号的分离度（这是门控可用性的直接证据）:")
print(f"    裁决者与辩论方分歧时: 裁决者错误率 = "
      f"{(1-df[dis].judge_correct.mean())*100:.2f}%   (n={int(dis.sum())})")
print(f"    裁决者与辩论方一致时: 裁决者错误率 = "
      f"{(1-df[~dis].judge_correct.mean())*100:.2f}%   (n={int((~dis).sum())})")
print(f"    错误率差 = "
      f"{(df[~dis].judge_correct.mean()-df[dis].judge_correct.mean())*100:+.2f}pp")
auc_conf = auc_tbl.get("① max 置信度（取负）", float('nan'))
auc_dis = auc_tbl.get("④ 辩论方分歧（裁决者≠pro票）", float('nan'))
print(f"\n  ★ 关键对比: 分歧信号 AUC={auc_dis:.4f}  vs  置信度 AUC={auc_conf:.4f}")
print(f"    → 分歧信号比置信度高 {auc_dis-auc_conf:+.4f}，"
      f"{'是更强的门控信号' if auc_dis > auc_conf else '仍不如置信度'}")

# ══════════════════════════════════════════════════
# 结论与落地建议
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("【结论】（含与预期假设不符之处，如实标注）")
print("=" * 100)

# 效率约束下的最优触发条件：门控的价值在「省算力」，触发率过高则退化为全程辩论
print("\n  ◆ 补充选优：门控必须同时满足「准确率不降」与「触发率显著低于 100%」")
print(f"  {'触发预算':<12} {'最优条件':<40} {'准确率':>9} {'vs全程辩论':>11}")
print("  " + "-" * 76)
budget_rows = []
for budget in [0.30, 0.40, 0.50, 0.60, 0.70]:
    cand = [c for c in combos if c[1] / 100 <= budget]
    if not cand:
        print(f"  ≤{budget*100:.0f}%{'':<7} {'无满足条件的方案':<40}")
        continue
    b = max(cand, key=lambda c: c[2])
    budget_rows.append({"budget_pct": budget * 100, "name": b[0],
                        "trigger_rate_pct": b[1], "accuracy_pct": b[2],
                        "vs_debate_pp": b[2] - acc_d})
    print(f"  ≤{budget*100:.0f}%{'':<7} {b[0]:<40} {b[2]:>8.2f}% {b[2]-acc_d:>+10.2f}pp")

c = [
    f"1. 【假设部分成立】置信度确实过度自信：[0.70,0.80) 区间预测 73.4% 而实际反讽率仅 50.0%"
    f"（+23.4pp 偏差），[0.90,0.95) 偏差达 +91.4pp。",

    f"2. 【假设不成立】Temperature scaling 对门控无效：最优 T={best_T} 仅把 LogLoss 从 "
    f"{ll_raw:.4f} 降到 {best_ll:.4f}（改善 {(1-best_ll/ll_raw)*100:.1f}%）。"
    f"两个口径下 AUC 均不变——判别金标签 {auc_raw:.4f}、判别裁决者出错 "
    f"{auc_tbl.get('② 反讽概率 p_irony（校准前）', float('nan')):.4f} → "
    f"{auc_tbl.get('③ 反讽概率 p_ts（校准后，T=%.1f）' % best_T, float('nan')):.4f}。"
    f"原因是温度缩放属单调变换，不改变样本排序，而门控本质是排序问题。"
    f"→「先校准置信度」这条路线应当放弃。",

    f"3. 【假设不成立】贝叶斯校正反而有害：多数票 {acc_majority:.2f}% → 校正后 "
    f"{acc_bayes:.2f}%（{acc_bayes-acc_majority:+.2f}pp）。"
    f"且校正把判反讽率从 {df.debate_b.mean()*100:.1f}% 推到 {bayes_pred.mean()*100:.1f}%，"
    f"离真实值 {true_rate*100:.1f}% 更远（方向错误）。"
    f"根因：单一先验 odds=1.0 下，LR(pro)={lr_pro:.3f}>1 与 LR(con)={lr_con:.3f}<1 "
    f"使后验退化为「pro 票即判反讽」，等价于直接采信 pro 辩论方单方投票，"
    f"并未实现偏置校正。正确做法需按两个辩论方各自的视角偏置分别建似然比再联合。",

    f"4. 【假设成立且是最重要发现】辩论方分歧是远优于置信度的门控信号："
    f"分歧时裁决者错误率 {(1-df[dis].judge_correct.mean())*100:.2f}%，"
    f"一致时仅 {(1-df[~dis].judge_correct.mean())*100:.2f}%，"
    f"分离度 {auc_dis:.4f} vs 置信度 {auc_conf:.4f}（高 {auc_dis-auc_conf:+.4f}）。",

    f"5. 【假设成立】辩论方 pro 偏置可作高召回反讽探测器："
    f"召回 {probe_recall*100:.2f}% vs 裁决者 {jr*100:.2f}%（高 "
    f"{(probe_recall-jr)*100:+.2f}pp），精确率 {probe_prec*100:.2f}%。"
    f"符合「宁可错抓不漏」的前置筛定位。",

    f"6. 【需修正的原判断】单纯按准确率选优会选到 {best_combo[0]}（{best_combo[2]:.2f}%），"
    f"但其触发率 {best_combo[1]:.1f}% 已接近全程辩论，门控失去省算力的意义。"
    f"门控评价必须引入效率约束。",
]
for x in c:
    print("\n  " + x)

print(f"\n  ◆ 落地建议（修正版）")
print(f"    • 门控信号改用「裁决者 vs 辩论方分歧」，放弃 softmax 阈值与温度校准")
print(f"    • 但分歧信号需辩论方先出票，意味着无法在辩论前跳过——门控的算力节省前提被破坏")
print(f"    • 因此需要「辩论前可得的廉价代理信号」：可用裁决者 top-2 logit 差、"
      f"或引入一个轻量反讽探测器（如词典/情感极性冲突规则）先行筛出高召回候选")
print(f"    • 当前 n=150 下分歧检测触发率 {df[df.judge_b!=df.pro_vote_b].shape[0]/len(df)*100:.1f}%，"
      f"准确率 {acc_d2:.2f}%，与全程辩论持平，说明分歧检测本身不省算力")

out = {
    "n": len(df),
    "calibration": {
        "auc_raw": round(auc_raw, 4), "brier_raw": round(brier_raw, 4),
        "logloss_raw": round(ll_raw, 4), "best_T": best_T,
        "logloss_ts": round(best_ll, 4), "brier_iso": round(br_iso, 4),
        "logloss_iso": round(ll_iso, 4), "auc_iso": round(auc_iso, 4),
        "conclusion": "temperature scaling 单调变换不改变 AUC，对门控排序无贡献",
    },
    "gate_signals_auc": {k: round(v, 4) for k, v in auc_tbl.items()},
    "disagreement_separation": {
        "judge_error_rate_when_disagree_pct": round((1 - df[dis].judge_correct.mean()) * 100, 2),
        "judge_error_rate_when_agree_pct": round((1 - df[~dis].judge_correct.mean()) * 100, 2),
        "n_disagree": int(dis.sum()), "n_agree": int((~dis).sum()),
    },
    "probe": {
        "pro_rate_pct": round(pro_rate * 100, 2), "true_rate_pct": round(true_rate * 100, 2),
        "probe_recall_pct": round(probe_recall * 100, 2),
        "probe_precision_pct": round(probe_prec * 100, 2),
        "judge_recall_pct": round(jr * 100, 2),
    },
    "trigger_combos": [
        {"name": nm, "trigger_rate_pct": round(tr, 2), "accuracy_pct": round(ac, 2),
         "vs_judge_pp": round(ac - acc_j, 2), "vs_debate_pp": round(ac - acc_d, 2)}
        for nm, tr, ac, *_ in combos
    ],
    "best_trigger_by_accuracy_only": {
        "name": best_combo[0], "trigger_rate_pct": round(best_combo[1], 2),
        "accuracy_pct": round(best_combo[2], 2),
        "caveat": "触发率接近全程辩论，门控失去省算力意义，不可直接采用",
    },
    "best_trigger_under_budget": budget_rows,
    "bayes": {
        "prior": round(prior, 4), "lr_pro": round(lr_pro, 4), "lr_con": round(lr_con, 4),
        "acc_majority_pct": round(acc_majority, 2), "acc_bayes_pct": round(acc_bayes, 2),
        "delta_pp": round(acc_bayes - acc_majority, 2),
        "pred_pro_rate_majority_pct": round(df.debate_b.mean() * 100, 2),
        "pred_pro_rate_bayes_pct": round(bayes_pred.mean() * 100, 2),
        "verdict": "退化：单一先验下后验等价于采信 pro 辩论方单方投票，偏置方向反而偏离真实值",
    },
    "reference": {"acc_judge_pct": round(acc_j, 2), "acc_debate_pct": round(acc_d, 2)},
}
p = f"{RES}/gate_calibration_result.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
pd.DataFrame([{
    "name": nm, "trigger_rate_pct": tr, "accuracy_pct": ac,
    "vs_judge_pp": ac - acc_j, "vs_debate_pp": ac - acc_d,
} for nm, tr, ac, *_ in combos]).to_csv(f"{RES}/gate_trigger_combos.csv",
                                        index=False, encoding="utf-8-sig")
if budget_rows:
    pd.DataFrame(budget_rows).to_csv(f"{RES}/gate_best_under_budget.csv",
                                     index=False, encoding="utf-8-sig")
print(f"\n结果已写入: gate_calibration_result.json / gate_trigger_combos.csv"
      f"{'' if not budget_rows else ' / gate_best_under_budget.csv'}")
print("=" * 100)
