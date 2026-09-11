"""门控与全票制冲突的解法设计与实证。

冲突机理（audit_gate_effectiveness.py 实测）：
  门控把高置信样本交还【裁决者单独判决】，而全票制在这些样本上本就优于裁决者，
  故门控等于撤销了全票制的收益 —— 两者争夺同一批样本的决策权（2/3 数据集组合更差）。

解法思路：不要让两者作用于同一决策维度。
  门控原本回答「是否辩论」（二元）；改为回答「用哪条计票规则」（分区治理）。
  这样高置信区、中置信区、低置信区各用最适合的规则，不再互相撤销。

实证四种解法（全部用已落盘数据，纯 CPU，不重跑推理）：
  S0 现状：门控(跳过) + 多数票            ← 冲突基线
  S1 分区计票：按置信度分三区，各区用不同票数门槛
  S2 二维联合标定：网格搜索 (τ, vote_threshold) 最优组合
  S3 非对称门控：只在「裁决者判 con 且低置信」时辩论（利用辩论方 pro 偏置补漏判）
  S4 融合式门控（对应论文模块 IV）：门控输出作为融合权重，而非硬跳过

关键纪律：所有方案的参数必须在【一个数据集上拟合、在其余数据集上验证】，
否则就是 in-sample 过拟合（本项目已因此推翻过事后分层估算）。
"""
import os
import json
import math
import numpy as np
import pandas as pd

RES = "experiments/results"
V2B = {"pro": 1, "con": 0}

SETS = [
    ("n=150 均衡", f"{RES}/debate_lens_n150_with.csv"),
    ("n=200 自然", f"{RES}/natural200_debate_with.csv"),
    ("n=784 中期", f"{RES}/roberta_B_full784_debate_with.partial.csv"),
]


def load(path):
    if not os.path.exists(path):
        return None
    try:
        d = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return None
    d = d.dropna(subset=["pro_vote", "con_vote", "predicted", "judge_vote"]).copy()
    d["y"] = d.true_label.astype(int)
    d["pv"] = d.pro_vote.map(V2B)
    d["cv"] = d.con_vote.map(V2B)
    d["jv"] = d.judge_vote.map(V2B)
    d["conf"] = pd.to_numeric(d.judge_confidence, errors="coerce")
    d = d.dropna(subset=["conf"]).reset_index(drop=True)
    d["vsum"] = d.pv + d.cv + d.jv
    return d


print("=" * 108)
print("门控 × 全票制 冲突解法实证")
print("=" * 108)
data = {}
for name, path in SETS:
    d = load(path)
    if d is None or len(d) < 30:
        print(f"  [跳过] {name}")
        continue
    data[name] = d
    print(f"  [载入] {name:<14} n={len(d):>4}  正类占比 {d.y.mean()*100:>5.1f}%  "
          f"置信度中位 {d.conf.median():.3f}  p90 {d.conf.quantile(.90):.3f}")


def acc(pred, y):
    return (np.asarray(pred) == np.asarray(y)).mean() * 100


# ══════════════════════════════════════════════════
# S0 冲突基线
# ══════════════════════════════════════════════════
print("\n" + "=" * 108)
print("【S0】冲突基线：门控(硬跳过) + 多数票 / 全票制")
print("=" * 108)
print(f"  {'数据集':<14} {'纯裁决者':>10} {'多数票':>9} {'全票制':>9} "
      f"{'门控0.90+多数票':>16} {'门控0.90+全票制':>16}")
print("  " + "-" * 80)
s0 = []
for k, d in data.items():
    y = d.y.values
    aj = acc(d.jv.values, y)
    amaj = acc((d.vsum.values >= 2).astype(int), y)
    aun = acc((d.vsum.values >= 3).astype(int), y)
    skip = d.conf.values >= 0.90
    a_g_maj = acc(np.where(skip, d.jv.values, (d.vsum.values >= 2).astype(int)), y)
    a_g_un = acc(np.where(skip, d.jv.values, (d.vsum.values >= 3).astype(int)), y)
    s0.append({"set": k, "judge": aj, "majority": amaj, "unanimity": aun,
               "gate_majority": a_g_maj, "gate_unanimity": a_g_un,
               "conflict_pp": round(a_g_un - aun, 2)})
    print(f"  {k:<14} {aj:>9.2f}% {amaj:>8.2f}% {aun:>8.2f}% "
          f"{a_g_maj:>15.2f}% {a_g_un:>15.2f}%")
print(f"\n  → 「门控+全票制」vs「纯全票制」: "
      + "，".join(f"{r['set']} {r['conflict_pp']:+.2f}pp" for r in s0))

# ══════════════════════════════════════════════════
# S1 分区计票
# ══════════════════════════════════════════════════
print("\n" + "=" * 108)
print("【S1】分区计票：门控不再「跳过辩论」，而是「选择计票规则」")
print("=" * 108)
print("""  设计：按裁决者置信度分三区，每区用不同的票数门槛
    高置信区 (conf ≥ τ_hi) : 裁决者已很确定 → 用【全票制 ≥3】（最严，避免辩论方偏置污染）
    中置信区 (τ_lo ≤ conf < τ_hi) : 用【多数票 ≥2】（标准）
    低置信区 (conf < τ_lo) : 裁决者不可靠 → 用【≥1 票】（最宽松，让辩论方主导，
                             利用其高召回特性补裁决者的漏判）
  与 S0 的本质区别：所有样本【都辩论】，门控只决定计票严格度 → 不丢任何辩论数据，
                    不与全票制争夺决策权（全票制成为高置信区的规则而非全局规则）""")

ZONES = [
    ("Z1 高置信全票/中置信多数/低置信≥1", 0.90, 0.70, 3, 2, 1),
    ("Z2 高置信全票/其余多数", 0.90, 0.00, 3, 2, 2),
    ("Z3 高置信全票/中置信≥1/低置信≥1", 0.90, 0.70, 3, 1, 1),
    ("Z4 高置信多数/中置信全票/低置信≥1", 0.90, 0.70, 2, 3, 1),
    ("Z5 全部≥1（辩论方主导）", 1.01, 1.01, 1, 1, 1),
]


def zone_predict(d, tau_hi, tau_lo, k_hi, k_mid, k_lo):
    conf = d.conf.values
    vsum = d.vsum.values
    pred = np.empty(len(d), dtype=int)
    hi = conf >= tau_hi
    lo = conf < tau_lo
    mid = ~hi & ~lo
    pred[hi] = (vsum[hi] >= k_hi).astype(int)
    pred[mid] = (vsum[mid] >= k_mid).astype(int)
    pred[lo] = (vsum[lo] >= k_lo).astype(int)
    return pred


print(f"\n  {'方案':<40} " + " ".join(f"{k:>13}" for k in data))
print("  " + "-" * (42 + 14 * len(data)))
s1 = []
for zname, thi, tlo, khi, kmid, klo in ZONES:
    vals, gains = [], []
    for k, d in data.items():
        y = d.y.values
        pred = zone_predict(d, thi, tlo, khi, kmid, klo)
        a = acc(pred, y)
        aj = acc(d.jv.values, y)
        vals.append(f"{a:.2f}%")
        gains.append(a - aj)
    zone_dist = []
    for k, d in data.items():
        hi = (d.conf.values >= thi).sum()
        lo = (d.conf.values < tlo).sum()
        zone_dist.append(f"{k}:高{hi}/低{lo}")
    s1.append({"plan": zname, "tau_hi": thi, "tau_lo": tlo,
               "k_hi": khi, "k_mid": kmid, "k_lo": klo,
               "acc_by_set": dict(zip(data.keys(), [float(v.rstrip('%')) for v in vals])),
               "gain_by_set": [round(g, 2) for g in gains],
               "all_positive": all(g > 0 for g in gains)})
    print(f"  {zname:<40} " + " ".join(f"{v:>13}" for v in vals))
print(f"\n  各区样本数: " + " | ".join(zone_dist))
best_s1 = max(s1, key=lambda r: min(r["gain_by_set"]))
print(f"\n  ★ 按【最差数据集增益】选优（而非平均，避免跨集平均谬误）: {best_s1['plan']}")
print(f"    各集增益: " + "，".join(f"{k} {g:+.2f}pp"
                                  for k, g in zip(data.keys(), best_s1["gain_by_set"])))

# ══════════════════════════════════════════════════
# S2 二维联合标定 + 跨集验证（关键纪律）
# ══════════════════════════════════════════════════
print("\n" + "=" * 108)
print("【S2】二维联合标定 (τ, 票数门槛) —— 含跨集验证防过拟合")
print("=" * 108)
print("""  纪律：在 n=150 上网格搜索最优 (τ, k)，然后【原样应用】到 n=200 与 n=784 验证。
  这是本项目已学到的教训 —— 事后分层估算因缺乏跨集验证而被实测推翻（估算 +13.43pp
  vs 实测 +2.00pp），in-sample 选参的数字不可引用。""")

TAUS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
KS = [1, 2, 3]

fit_name = "n=150 均衡"
if fit_name in data:
    fd = data[fit_name]
    fy = fd.y.values
    grid = []
    for tau in TAUS:
        for kk in KS:
            skip = fd.conf.values >= tau
            pred = np.where(skip, fd.jv.values, (fd.vsum.values >= kk).astype(int))
            a = acc(pred, fy)
            grid.append({"tau": tau, "k": kk, "fit_acc": round(a, 2),
                         "debate_rate_pct": round((~skip).mean() * 100, 2)})
    grid.sort(key=lambda r: -r["fit_acc"])
    print(f"\n  拟合集 {fit_name} 上的 Top-8 组合:")
    print(f"  {'τ':>6} {'门槛':>6} {'拟合准确率':>11} {'辩论触发率':>11}")
    print("  " + "-" * 40)
    for r in grid[:8]:
        print(f"  {r['tau']:>6.2f} {'≥'+str(r['k']):>6} {r['fit_acc']:>10.2f}% "
              f"{r['debate_rate_pct']:>10.1f}%")

    best = grid[0]
    print(f"\n  选定 (τ={best['tau']:.2f}, 门槛≥{best['k']})，在未见数据集上验证:")
    print(f"  {'验证集':<14} {'纯裁决者':>10} {'多数票':>9} {'全票制':>9} "
          f"{'选定组合':>10} {'vs裁决者':>10} {'vs全票制':>10}")
    print("  " + "-" * 78)
    s2 = []
    for k, d in data.items():
        y = d.y.values
        aj = acc(d.jv.values, y)
        amaj = acc((d.vsum.values >= 2).astype(int), y)
        aun = acc((d.vsum.values >= 3).astype(int), y)
        skip = d.conf.values >= best["tau"]
        pred = np.where(skip, d.jv.values, (d.vsum.values >= best["k"]).astype(int))
        a = acc(pred, y)
        tag = "(拟合集)" if k == fit_name else ""
        s2.append({"set": k, "is_fit_set": k == fit_name, "acc_judge": round(aj, 2),
                   "acc_majority": round(amaj, 2), "acc_unanimity": round(aun, 2),
                   "acc_selected": round(a, 2), "vs_judge_pp": round(a - aj, 2),
                   "vs_unanimity_pp": round(a - aun, 2),
                   "debate_rate_pct": round((~skip).mean() * 100, 2)})
        print(f"  {k+tag:<14} {aj:>9.2f}% {amaj:>8.2f}% {aun:>8.2f}% "
              f"{a:>9.2f}% {a-aj:>+9.2f}pp {a-aun:>+9.2f}pp")
    unseen = [r for r in s2 if not r["is_fit_set"]]
    if unseen:
        ok_j = sum(1 for r in unseen if r["vs_judge_pp"] > 0)
        ok_u = sum(1 for r in unseen if r["vs_unanimity_pp"] > 0)
        print(f"\n  → 未见集上优于纯裁决者: {ok_j}/{len(unseen)}   "
              f"优于纯全票制: {ok_u}/{len(unseen)}")
        if ok_u < len(unseen):
            print(f"  ★ 泛化失败：选定组合在未见集上不如【简单的全票制】，")
            print(f"    说明二维标定过拟合拟合集的置信度分布。")
            print(f"    → 应优先用全票制（无超参、跨集稳定），而非联合标定。")
        else:
            print(f"  ★ 泛化成立：联合标定优于全票制，可作为部署配置。")
else:
    s2, grid, best = [], [], None
    print(f"  [跳过] 未找到拟合集 {fit_name}")

# ══════════════════════════════════════════════════
# S3 非对称门控
# ══════════════════════════════════════════════════
print("\n" + "=" * 108)
print("【S3】非对称门控：只在「裁决者判 con 且低置信」时辩论")
print("=" * 108)
print("""  依据（diagnose_lens_inversion.py 实测）：
    裁决者偏置为【负】（倾向判 con / 漏判正类），辩论方偏置为【正】（倾向判 pro）
    → 辩论的价值集中在「裁决者判 con」的子集（补漏判）
    → 在「裁决者判 pro」的子集上，辩论只会强化既有的 pro 偏置，无纠错价值
  故门控条件改为：jv==con AND conf<τ → 辩论；否则采信裁决者""")
print(f"\n  {'数据集':<14} {'τ':>5} {'辩论触发率':>11} {'准确率':>9} "
      f"{'vs裁决者':>10} {'vs全票制':>10}")
print("  " + "-" * 66)
s3 = []
for k, d in data.items():
    y = d.y.values
    aj = acc(d.jv.values, y)
    aun = acc((d.vsum.values >= 3).astype(int), y)
    for tau in [0.70, 0.80, 0.90, 0.95, 1.01]:
        need = (d.jv.values == 0) & (d.conf.values < tau)
        pred = np.where(need, (d.vsum.values >= 2).astype(int), d.jv.values)
        a = acc(pred, y)
        s3.append({"set": k, "tau": tau, "trigger_pct": round(need.mean() * 100, 2),
                   "accuracy_pct": round(a, 2), "vs_judge_pp": round(a - aj, 2),
                   "vs_unanimity_pp": round(a - aun, 2)})
        print(f"  {k:<14} {tau:>5.2f} {need.mean()*100:>10.1f}% {a:>8.2f}% "
              f"{a-aj:>+9.2f}pp {a-aun:>+9.2f}pp")
    print("  " + "-" * 66)

# ══════════════════════════════════════════════════
# S4 融合式门控（对应论文模块 IV）
# ══════════════════════════════════════════════════
print("\n" + "=" * 108)
print("【S4】融合式门控（对应论文模块 IV「跨模态门控融合」的文本版）")
print("=" * 108)
print("""  本质区别：门控输出不再做【硬跳过】(0/1)，而是作为【连续融合权重】 g∈[0,1]：
      final_score = g × 裁决者分数 + (1−g) × 辩论方分数
      g = 裁决者置信度（或其校准版本）
  这样门控与计票规则不再争夺决策权，而是各自贡献一个连续量 → 结构上消除冲突。
  这也正是论文图 2.2 的「门控融合」语义，可直接迁移到多模态（图像门控 + 文本门控）。""")

print(f"\n  {'数据集':<14} {'方案':<34} {'准确率':>9} {'vs裁决者':>10} {'vs全票制':>10}")
print("  " + "-" * 82)
s4 = []
for k, d in data.items():
    y = d.y.values
    aj = acc(d.jv.values, y)
    aun = acc((d.vsum.values >= 3).astype(int), y)
    conf = d.conf.values
    # 裁决者分数：判 pro 时为 conf，判 con 时为 1-conf
    s_judge = np.where(d.jv.values == 1, conf, 1 - conf)
    # 辩论方分数：投 pro 的票数比例
    s_debate = (d.pv.values + d.cv.values) / 2.0
    plans = [
        ("g=conf 线性融合", conf * s_judge + (1 - conf) * s_debate),
        ("g=conf² (偏向裁决者)", (conf ** 2) * s_judge + (1 - conf ** 2) * s_debate),
        ("g=√conf (偏向辩论方)", np.sqrt(conf) * s_judge + (1 - np.sqrt(conf)) * s_debate),
        ("g=0.5 等权融合", 0.5 * s_judge + 0.5 * s_debate),
    ]
    for pname, score in plans:
        pred = (score >= 0.5).astype(int)
        a = acc(pred, y)
        s4.append({"set": k, "plan": pname, "accuracy_pct": round(a, 2),
                   "vs_judge_pp": round(a - aj, 2),
                   "vs_unanimity_pp": round(a - aun, 2),
                   "pred_pro_rate_pct": round(pred.mean() * 100, 2)})
        print(f"  {k:<14} {pname:<34} {a:>8.2f}% {a-aj:>+9.2f}pp {a-aun:>+9.2f}pp")
    print("  " + "-" * 82)

# 融合式的最优阈值也做跨集验证
print(f"\n  ◆ 融合分数的判决阈值同样需跨集标定（阈值≠0.5 时）")
if fit_name in data:
    fd = data[fit_name]
    conf_f = fd.conf.values
    sj = np.where(fd.jv.values == 1, conf_f, 1 - conf_f)
    sd = (fd.pv.values + fd.cv.values) / 2.0
    score_f = conf_f * sj + (1 - conf_f) * sd
    thr_grid = [(t, acc((score_f >= t).astype(int), fd.y.values))
                for t in np.arange(0.35, 0.72, 0.025)]
    thr_best, acc_best = max(thr_grid, key=lambda x: x[1])
    print(f"    拟合集 {fit_name}: 最优阈值 {thr_best:.3f} → {acc_best:.2f}%")
    thr_label = f"阈值{thr_best:.3f}"
    print(f"    {'验证集':<14} {'阈值0.50':>10} {thr_label:>12} "
          f"{'纯裁决者':>10} {'全票制':>9}")
    print("    " + "-" * 60)
    s4_thr = []
    for k, d in data.items():
        y = d.y.values
        c = d.conf.values
        sj2 = np.where(d.jv.values == 1, c, 1 - c)
        sd2 = (d.pv.values + d.cv.values) / 2.0
        sc = c * sj2 + (1 - c) * sd2
        a50 = acc((sc >= 0.5).astype(int), y)
        ab = acc((sc >= thr_best).astype(int), y)
        s4_thr.append({"set": k, "is_fit": k == fit_name,
                       "acc_thr050": round(a50, 2), "acc_thr_best": round(ab, 2),
                       "acc_judge": round(acc(d.jv.values, y), 2),
                       "acc_unanimity": round(acc((d.vsum.values >= 3).astype(int), y), 2)})
        print(f"    {k+('(拟合)' if k==fit_name else ''):<14} {a50:>9.2f}% {ab:>11.2f}% "
              f"{acc(d.jv.values,y):>9.2f}% "
              f"{acc((d.vsum.values>=3).astype(int),y):>8.2f}%")
else:
    s4_thr = []

# ══════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════
print("\n" + "=" * 108)
print("【汇总】四种解法的对比（按「最差数据集增益」排序，避免跨集平均谬误）")
print("=" * 108)
allplans = []
for r in s0:
    allplans.append({"family": "S0 冲突基线", "plan": f"门控0.90+全票制 @ {r['set']}",
                     "set": r["set"], "gain": r["conflict_pp"]})
for r in s1:
    for k, g in zip(data.keys(), r["gain_by_set"]):
        allplans.append({"family": "S1 分区计票", "plan": r["plan"], "set": k, "gain": g})
for r in s3:
    allplans.append({"family": "S3 非对称门控", "plan": f"τ={r['tau']:.2f}",
                     "set": r["set"], "gain": r["vs_judge_pp"]})
for r in s4:
    allplans.append({"family": "S4 融合式门控", "plan": r["plan"],
                     "set": r["set"], "gain": r["vs_judge_pp"]})
# 全票制自身
for k, d in data.items():
    allplans.append({"family": "参照 全票制", "plan": "全程辩论+全票制≥3", "set": k,
                     "gain": round(acc((d.vsum.values >= 3).astype(int), d.y.values)
                                   - acc(d.jv.values, d.y.values), 2)})

dfp = pd.DataFrame(allplans)
print(f"  {'方案族':<18} {'方案':<40} {'最差集增益':>11} {'各集增益':>26}")
print("  " + "-" * 100)
ranked = []
for (fam, plan), grp in dfp.groupby(["family", "plan"]):
    if len(grp) < len(data):
        continue
    worst = grp.gain.min()
    ranked.append({"family": fam, "plan": plan, "worst_gain_pp": round(float(worst), 2),
                   "gains": {r["set"]: r["gain"] for _, r in grp.iterrows()},
                   "all_positive": bool((grp.gain > 0).all())})
ranked.sort(key=lambda r: -r["worst_gain_pp"])
for r in ranked[:14]:
    gs = "，".join(f"{v:+.1f}" for v in r["gains"].values())
    star = " ★" if r["all_positive"] else ""
    print(f"  {r['family']:<18} {r['plan']:<40} {r['worst_gain_pp']:>+10.2f}pp "
          f"{gs:>26}{star}")

top = ranked[0] if ranked else None
print(f"\n  ★ 最差数据集上增益最高的方案: {top['family']} / {top['plan']}  "
      f"({top['worst_gain_pp']:+.2f}pp)" if top else "")
pos_only = [r for r in ranked if r["all_positive"]]
if pos_only:
    print(f"  ★ 在【全部】数据集上均为正增益的方案（稳健解）:")
    for r in pos_only[:6]:
        gs = "，".join(f"{k} {v:+.2f}pp" for k, v in r["gains"].items())
        print(f"      {r['family']} / {r['plan']:<38} {gs}")

out = {
    "S0_conflict_baseline": s0,
    "S1_zone_voting": s1,
    "S2_grid_search": {"grid_top": grid[:8], "selected": best, "validation": s2},
    "S3_asymmetric_gate": s3,
    "S4_fusion_gate": s4,
    "S4_threshold_crossval": s4_thr,
    "ranking_by_worst_case": ranked,
    "robust_plans": pos_only,
    "conflict_resolution_verdict": (
        "冲突根因是门控与全票制争夺同一批样本的决策权；"
        "解法是让门控从「硬跳过」改为「选择计票规则」(S1) 或「连续融合权重」(S4)，"
        "使两者作用于不同维度"),
}
with open(f"{RES}/gate_conflict_resolution.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
dfp.to_csv(f"{RES}/gate_resolution_all_plans.csv", index=False, encoding="utf-8-sig")
print(f"\n结果已写入: gate_conflict_resolution.json / gate_resolution_all_plans.csv")
print("=" * 108)
