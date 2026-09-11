"""中期配对分析：用已落盘的 partial 辩论结果 vs 同批样本的裁决者基线。

为什么能提前判断：
  partial CSV（每 25 条增量落盘）与 conf_{judge}_semeval_test.csv 都按 irony_test.csv
  原始行序生成，故可按 text 对齐，对【同一批已完成样本】做配对比较。
  这样不必等 5.7h 跑完就能看出辩论增益的方向。

口径：McNemar 配对检验 + 分类别命中 + 门控扫描，与最终脚本一致。
纯 CPU，不占用 GPU。
"""
import os
import sys
import math
import numpy as np
import pandas as pd

RES = "experiments/results"
OUT = sys.argv[1] if len(sys.argv) > 1 else "roberta_B_full784"
JUDGE = sys.argv[2] if len(sys.argv) > 2 else "judge_roberta_cw_samedomain_ablation"

partial = f"{RES}/{OUT}_debate_with.partial.csv"
final = f"{RES}/{OUT}_debate_with.csv"
conf = f"{RES}/conf_{JUDGE}_semeval_test.csv"

src = final if os.path.exists(final) else partial
if not os.path.exists(src):
    raise SystemExit(f"[FAIL] 未找到 {partial} 或 {final}，实验可能尚未产出")

try:
    d = pd.read_csv(src)
except pd.errors.ParserError:
    # 写入瞬间读到半行，退一行重读
    d = pd.read_csv(src, on_bad_lines="skip")

if not os.path.exists(conf):
    raise SystemExit(f"[FAIL] 未找到裁决者置信度文件 {conf}")
c = pd.read_csv(conf)

stage = "最终结果" if src == final else "中期 partial"
print("=" * 100)
print(f"辩论 vs 裁决者 配对分析【{stage}】")
print(f"  辩论数据: {os.path.basename(src)}   已完成 {len(d)} 条")
print(f"  裁决者基线: {os.path.basename(conf)}   共 {len(c)} 条")
print("=" * 100)

# ── 对齐校验：必须按 text 确认同序，否则配对无意义 ──
d = d.reset_index(drop=True)
n = len(d)
if n > len(c):
    raise SystemExit(f"[FAIL] 辩论条数 {n} 超过基线 {len(c)}，无法对齐")

sub = c.iloc[:n].reset_index(drop=True)
dt = d["text"].astype(str).str.strip()
ct = sub["text"].astype(str).str.strip()
mismatch = int((dt != ct).sum())
print(f"\n【对齐校验】前 {n} 条 text 逐行比对: 不匹配 {mismatch} 条")
if mismatch > 0:
    print("  [FAIL] 顺序不一致，配对分析无效，改用 text 连接")
    m = d.merge(c[["text", "predicted", "confidence", "true_label"]],
                on="text", how="inner", suffixes=("", "_j"))
    if len(m) < n * 0.9:
        raise SystemExit("  连接后样本损失过大，放弃")
    d = m.reset_index(drop=True)
    n = len(d)
    sub = None
    print(f"  已改用 text 连接，有效配对 {n} 条")
else:
    print("  [OK] 顺序一致，可按行配对")

# ── 统一口径 ──
y = d["true_label"].astype(int).values
deb = d["predicted"].astype(int).values if pd.api.types.is_numeric_dtype(d["predicted"]) \
    else d["predicted"].map({"pro": 1, "con": 0}).astype(int).values
if sub is not None:
    jud = sub["predicted"].astype(int).values
    jconf = sub["confidence"].values
else:
    jud = d["predicted_j"].map({"pro": 1, "con": 0}).astype(int).values \
        if not pd.api.types.is_numeric_dtype(d["predicted_j"]) \
        else d["predicted_j"].astype(int).values
    jconf = d["confidence"].values

# 交叉校验：辩论 CSV 里自带的 judge_vote 应与基线一致（同模型同样本）
if "judge_vote" in d.columns:
    jv = d["judge_vote"].map({"pro": 1, "con": 0}).astype(int).values
    agree = (jv == jud).mean() * 100
    print(f"【一致性校验】辩论CSV内judge_vote vs 基线predicted 一致率: {agree:.2f}%")
    if agree < 99:
        print("  [WARN] 不一致，可能裁决者模型或解码设置不同，结论需谨慎")
if "judge_confidence" in d.columns:
    cd = np.corrcoef(d["judge_confidence"].values, jconf)[0, 1]
    print(f"【置信度校验】两来源 confidence 相关系数: {cd:.6f}")

print(f"\n样本构成: 反讽 {int(y.sum())} / 非反讽 {int((y==0).sum())} "
      f"（反讽占比 {y.mean()*100:.1f}%）")


def se(pct, m):
    return math.sqrt(pct * (100 - pct) / m) if 0 < pct < 100 else 0.0


acc_d = (deb == y).mean() * 100
acc_j = (jud == y).mean() * 100
delta = acc_d - acc_j

ok_d = deb == y
ok_j = jud == y
b = int((~ok_j & ok_d).sum())    # 裁决者错、辩论对
cc = int((ok_j & ~ok_d).sum())   # 裁决者对、辩论错
chi2 = ((abs(b - cc) - 1) ** 2 / (b + cc)) if (b + cc) else 0.0
se_pair = (math.sqrt((b + cc) - (b - cc) ** 2 / (b + cc)) / n * 100) if (b + cc) else 0.0

print("\n" + "-" * 100)
print("① 准确率对照（同批样本，配对）")
print("-" * 100)
print(f"   视角分化辩论 : {acc_d:>7.2f}%  ±{se(acc_d, n)*1.96:.2f}pp")
print(f"   纯裁决者     : {acc_j:>7.2f}%  ±{se(acc_j, n)*1.96:.2f}pp")
print(f"   辩论增益     : {delta:>+7.2f}pp  ±{se_pair*1.96:.2f}pp")
print(f"   McNemar b={b} c={cc} χ²={chi2:.3f} → "
      f"{'显著' if chi2 > 3.841 else '不显著'}（临界 3.841, α=0.05）")
print(f"   不一致样本 {b+cc} 条（{(b+cc)/n*100:.1f}%）")

print("\n" + "-" * 100)
print("② 分类别命中")
print("-" * 100)
print(f"   {'类别':<12} {'裁决者':>9} {'辩论':>9} {'差值':>10} {'样本数':>7}")
cls = []
for lab, name in [(1, "真反讽"), (0, "真非反讽")]:
    m_ = y == lab
    aj = ok_j[m_].mean() * 100
    ad = ok_d[m_].mean() * 100
    cls.append((name, int(m_.sum()), aj, ad, ad - aj))
    print(f"   {name:<12} {aj:>8.2f}% {ad:>8.2f}% {ad-aj:>+9.2f}pp {int(m_.sum()):>7}")

g = cls[0][4]
h = cls[1][4]
if g - h != 0:
    pi_star = -h / (g - h) * 100
    print(f"\n   临界反讽占比 π* = {pi_star:.2f}%   本批实际 {y.mean()*100:.1f}%")
    print(f"   → 本批{'高于' if y.mean()*100 > pi_star else '低于'}临界值，"
          f"预期净增益{'为正' if y.mean()*100 > pi_star else '为负'}")
    print(f"   增益分解: π·Δ反讽 = {y.mean()*g:+.2f}pp, "
          f"(1-π)·Δ非反讽 = {(1-y.mean())*h:+.2f}pp, 合计 {y.mean()*g+(1-y.mean())*h:+.2f}pp")

print("\n" + "-" * 100)
print("③ 偏置与互补性")
print("-" * 100)
bd = deb.mean() * 100 - y.mean() * 100
bj = jud.mean() * 100 - y.mean() * 100
print(f"   真实反讽率 {y.mean()*100:.1f}%   辩论判pro {deb.mean()*100:.1f}%   "
      f"裁决者判pro {jud.mean()*100:.1f}%")
print(f"   辩论方 pro 偏置 {bd:+.2f}pp   裁决者 pro 偏置 {bj:+.2f}pp")
print(f"   互补性: {'反向（互补保留）' if bd*bj < 0 else '同向（叠加，辩论无纠正作用）'}")
if "pro_vote" in d.columns and "con_vote" in d.columns:
    pv = d["pro_vote"].map({"pro": 1, "con": 0})
    cv = d["con_vote"].map({"pro": 1, "con": 0})
    print(f"   正方投pro率 {pv.mean()*100:.1f}%   反方投pro率 {cv.mean()*100:.1f}%   "
          f"双方一致率 {(pv==cv).mean()*100:.1f}%")

print("\n" + "-" * 100)
print("④ 门控阈值扫描（本批样本，离线）")
print("-" * 100)
lat = d["latency_sec"].mean() if "latency_sec" in d.columns else 21.0
print(f"   {'τ':>6} {'准确率':>9} {'触发率':>8} {'vs裁决者':>10} {'vs全程辩论':>11} {'延迟节省':>9}")
best = None
for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]:
    low = jconf < tau
    pred = np.where(low, deb, jud)
    a = (pred == y).mean() * 100
    trig = low.mean() * 100
    save = trig / 100 * 0 + (1 - (trig / 100 * lat + (1 - trig / 100) * 0.01) / lat) * 100
    print(f"   {tau:>6.2f} {a:>8.2f}% {trig:>7.1f}% {a-acc_j:>+9.2f}pp "
          f"{a-acc_d:>+10.2f}pp {save:>8.0f}%")
    if trig <= 60.0 and (best is None or a > best[1]):
        best = (tau, a, trig)
if best:
    print(f"   ★ 触发率≤60% 内最优 τ={best[0]:.2f}: {best[1]:.2f}%  "
          f"vs裁决者 {best[1]-acc_j:+.2f}pp  vs全程辩论 {best[1]-acc_d:+.2f}pp")

# ── 统计功效：当前 n 能检出多大增益 ──
print("\n" + "-" * 100)
print("⑤ 当前样本量的统计功效")
print("-" * 100)
p_disc = (b + cc) / n


def power_for(d_pp, m, p_d):
    if p_d <= 0:
        return float("nan"), None
    p_b = 0.5 + (d_pp / 100) / (2 * p_d)
    if not (0 < p_b < 1):
        return float("nan"), None
    from statistics import NormalDist
    z = (p_b - 0.5) * math.sqrt(m * p_d) / 0.5
    return 1 - NormalDist().cdf(1.96 - z), ((1.96 + 0.8416) * 0.5 / (p_b - 0.5)) ** 2 / p_d


for dpp, lab in [(delta, "实测增益"), (4.50, "n=200门控增益"), (13.43, "事后分层估算"),
                 (18.67, "均衡集增益")]:
    pw, need = power_for(dpp, n, p_disc)
    ns = f"{need:>9.0f}" if need else "      不可达"
    print(f"   {lab:<18} {dpp:>+7.2f}pp   当前n={n}功效 {pw*100:>6.1f}%   "
          f"80%功效需n {ns}")

print("\n" + "=" * 100)
print(f"【判读】{stage}（n={n}）")
print("=" * 100)
if chi2 > 3.841 and delta > 0:
    v = "辩论增益为正且显著 → 框架成立，需跑 C/D 对照并上门控"
elif chi2 > 3.841 and delta < 0:
    v = "★ 辩论增益为负且显著 → 辩论有害，须停用或修辩论方偏置"
elif delta < -1.0:
    v = "辩论增益为负但不显著 → 倾向有害，需全量784条确认"
elif abs(delta) <= 1.0:
    v = "辩论增益≈0 → 辩论未带来净收益，门控与成本成为主要论点"
else:
    v = "辩论增益为正但不显著 → 功效不足，需更大样本"
print(f"   {v}")
if src == partial:
    print(f"   注意：这是 partial 中期结果，样本为 CSV 前 {n} 条（按原序，非随机抽样），"
          f"最终结论以全量 784 条为准。")
else:
    print("   注意：已是全量最终结果。")
print("=" * 100)
