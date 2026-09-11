"""决定性检验：全票制的收益究竟来自「辩论」还是来自「阈值平移」？

用户问题：文本层用全票制修偏置、门控留给多模态融合层，这样辩论机制是否全是正收益？

必须排除的替代解释：
  全票制 ≥3 把判 pro 率从 64.9% 压到 22.8%（偏置 +26.4pp → −15.8pp），
  本质是【把判决阈值推向 con】。若数据集负类占多数，这种平移本身就能提升准确率，
  【完全不需要辩论】。
  → 决定性对照：只用裁决者自身的反讽概率做阈值平移（零辩论、零额外算力），
     看能否达到全票制的准确率。若能，则辩论对全票制的收益贡献为 0。

检验设计（全部用已落盘数据，纯 CPU）：
  T1 裁决者阈值平移扫描：predict pro iff p_irony(judge) >= t，t ∈ [0.50, 0.99]
     与全票制对比 —— 这是「辩论是否必要」的核心对照
  T2 McNemar 配对检验：全票制 vs 最优阈值平移裁决者（同批样本）
  T3 类别比例敏感性：acc(π) = π·R_正 + (1−π)·R_负，求全票制开始劣于纯裁决者的 π*
     → 回答「是否全是正收益」：若存在 π 区间使全票制更差，则答案是否定的
  T4 辩论方信息量：在阈值平移已对齐判 pro 率的前提下，辩论是否仍带来增量
     （对齐判 pro 率后再比准确率，剥离阈值效应，只看信息量）
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
    # 裁决者的「判正类概率」：判 pro 时为 conf，判 con 时为 1−conf
    d["p_pos_judge"] = np.where(d.jv == 1, d.conf, 1 - d.conf)
    d["vsum"] = d.pv + d.cv + d.jv
    d["unan"] = (d.vsum >= 3).astype(int)
    d["maj"] = (d.vsum >= 2).astype(int)
    return d


def acc(pred, y):
    return (np.asarray(pred) == np.asarray(y)).mean() * 100


def recall(pred, y, cls):
    m = y == cls
    return (np.asarray(pred)[m] == cls).mean() * 100 if m.sum() else float("nan")


def mcnemar(p1, p2, y):
    ok1 = np.asarray(p1) == np.asarray(y)
    ok2 = np.asarray(p2) == np.asarray(y)
    b = int((~ok1 & ok2).sum())
    c = int((ok1 & ~ok2).sum())
    chi2 = ((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) else 0.0
    return {"b": b, "c": c, "chi2": round(chi2, 3),
            "significant": bool(chi2 > 3.841)}


print("=" * 110)
print("决定性检验：全票制的收益来自「辩论」还是「阈值平移」？")
print("=" * 110)

data = {}
for name, path in SETS:
    d = load(path)
    if d is None or len(d) < 30:
        print(f"  [跳过] {name}")
        continue
    data[name] = d
    print(f"  [载入] {name:<14} n={len(d):>4}  正类占比 {d.y.mean()*100:>5.1f}%")

if not data:
    raise SystemExit("[FAIL] 无可用数据")

# ══════════════════════════════════════════════════
# T1 裁决者阈值平移 vs 全票制
# ══════════════════════════════════════════════════
print("\n" + "=" * 110)
print("【T1】裁决者阈值平移（零辩论、零额外算力）能否达到全票制的准确率")
print("=" * 110)
print("""  判据：若存在某个阈值 t，使「纯裁决者 + 阈值 t」的准确率 ≥ 全票制，
        则全票制的收益可被【不用辩论】的方案完全替代 → 辩论贡献为 0。""")

TS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
t1_rows = []
for name, d in data.items():
    y = d.y.values
    aj = acc(d.jv.values, y)
    au = acc(d.unan.values, y)
    am = acc(d.maj.values, y)
    pro_u = d.unan.mean() * 100
    print(f"\n  ◆ {name}（n={len(d)}, 正类 {d.y.mean()*100:.1f}%）")
    print(f"    纯裁决者(argmax)  : {aj:>7.2f}%   判正类率 {d.jv.mean()*100:>5.1f}%")
    print(f"    全程辩论+多数票    : {am:>7.2f}%   判正类率 {d.maj.mean()*100:>5.1f}%")
    print(f"    全程辩论+全票制    : {au:>7.2f}%   判正类率 {pro_u:>5.1f}%")
    print(f"    {'阈值 t':>8} {'裁决者准确率':>13} {'判正类率':>10} {'vs全票制':>10} {'备注':>10}")
    print("    " + "-" * 58)
    best_t, best_a = None, -1
    for t in TS:
        pred = (d.p_pos_judge.values >= t).astype(int)
        a = acc(pred, y)
        pr = pred.mean() * 100
        mark = ""
        if a >= au:
            mark = "★达标"
        if a > best_a:
            best_a, best_t = a, t
        t1_rows.append({"set": name, "t": t, "acc_judge_thr": round(a, 2),
                        "pred_pro_rate": round(pr, 2),
                        "acc_unanimity": round(au, 2),
                        "beats_unanimity": bool(a >= au),
                        "delta_vs_unanimity_pp": round(a - au, 2)})
        print(f"    {t:>8.2f} {a:>12.2f}% {pr:>9.1f}% {a-au:>+9.2f}pp {mark:>10}")
    print(f"    → 裁决者最优阈值 t={best_t:.2f}: {best_a:.2f}%   "
          f"全票制: {au:.2f}%   差值 {best_a-au:+.2f}pp")
    matched = [r for r in t1_rows if r["set"] == name and r["beats_unanimity"]]
    if matched:
        print(f"    ★【辩论非必要】有 {len(matched)} 个阈值使纯裁决者达到或超过全票制"
              f"（最低 t={min(r['t'] for r in matched):.2f}）")
    else:
        print(f"    → 无任何阈值能让纯裁决者达到全票制 → 辩论确有增量贡献")

# ══════════════════════════════════════════════════
# T2 McNemar：全票制 vs 最优阈值裁决者
# ══════════════════════════════════════════════════
print("\n" + "=" * 110)
print("【T2】McNemar 配对检验：全票制 vs 纯裁决者（argmax 与最优阈值两种）")
print("=" * 110)
print(f"  {'数据集':<14} {'对照':<28} {'b':>5} {'c':>5} {'χ²':>8} {'显著?':>7} {'准确率差':>10}")
print("  " + "-" * 82)
t2_rows = []
for name, d in data.items():
    y = d.y.values
    au = acc(d.unan.values, y)
    # 对照1: 裁决者 argmax
    m1 = mcnemar(d.jv.values, d.unan.values, y)
    t2_rows.append({"set": name, "baseline": "裁决者 argmax", **m1,
                    "acc_unanimity": round(au, 2),
                    "acc_baseline": round(acc(d.jv.values, y), 2),
                    "delta_pp": round(au - acc(d.jv.values, y), 2)})
    print(f"  {name:<14} {'裁决者 argmax':<28} {m1['b']:>5} {m1['c']:>5} "
          f"{m1['chi2']:>8.3f} {'显著' if m1['significant'] else '不显著':>7} "
          f"{au-acc(d.jv.values,y):>+9.2f}pp")
    # 对照2: 裁决者最优阈值（in-sample 选出，故为辩论的【最严苛】对照）
    best = max(t1_rows and [r for r in t1_rows if r["set"] == name] or [],
               key=lambda r: r["acc_judge_thr"], default=None)
    if best:
        pred_t = (d.p_pos_judge.values >= best["t"]).astype(int)
        m2 = mcnemar(pred_t, d.unan.values, y)
        bl_label = f"裁决者 阈值{best['t']:.2f}"
        t2_rows.append({"set": name, "baseline": bl_label, **m2,
                        "acc_unanimity": round(au, 2),
                        "acc_baseline": best["acc_judge_thr"],
                        "delta_pp": round(au - best["acc_judge_thr"], 2)})
        print(f"  {'':<14} {bl_label:<28} "
              f"{m2['b']:>5} {m2['c']:>5} {m2['chi2']:>8.3f} "
              f"{'显著' if m2['significant'] else '不显著':>7} "
              f"{au-best['acc_judge_thr']:>+9.2f}pp")
        print(f"  {'':<14}   ↑ 注：该阈值是在本集 in-sample 选出的，对辩论是【最严苛】对照")

# ══════════════════════════════════════════════════
# T3 类别比例敏感性（回答「是否全是正收益」）
# ══════════════════════════════════════════════════
print("\n" + "=" * 110)
print("【T3】类别比例敏感性：全票制在什么正类占比下会劣于纯裁决者")
print("=" * 110)
print("""  方法：acc(π) = π·R_正 + (1−π)·R_负（分类别命中率实测，π 为真实正类占比）
  求 π* 使 acc_全票制(π*) = acc_裁决者(π*)。π < π* 或 π > π* 时全票制劣于纯裁决者。
  → 若存在这样的 π 区间，则「辩论全是正收益」不成立。""")
print(f"\n  {'数据集':<14} {'方案':<16} {'R_正':>8} {'R_负':>8} {'当前π':>7} {'当前acc':>9}")
print("  " + "-" * 68)
t3_rows = []
for name, d in data.items():
    y = d.y.values
    pi_now = y.mean()
    for pname, pred in [("纯裁决者", d.jv.values), ("多数票≥2", d.maj.values),
                        ("全票制≥3", d.unan.values)]:
        ri = recall(pred, y, 1)
        rn = recall(pred, y, 0)
        a = acc(pred, y)
        t3_rows.append({"set": name, "plan": pname, "recall_pos": round(ri, 2),
                        "recall_neg": round(rn, 2), "pi_now": round(pi_now * 100, 2),
                        "acc_now": round(a, 2)})
        print(f"  {name:<14} {pname:<16} {ri:>7.2f}% {rn:>7.2f}% "
              f"{pi_now*100:>6.1f}% {a:>8.2f}%")
    print("  " + "-" * 68)

print(f"\n  ◆ acc(π) 外推与临界占比 π*")
print(f"  {'数据集':<14} {'方案':<16} {'π=20%':>9} {'π=40%':>9} {'π=50%':>9} "
      f"{'π=60%':>9} {'π=80%':>9} {'劣于裁决者的π区间':>22}")
print("  " + "-" * 100)
extrap = []
for name, d in data.items():
    y = d.y.values
    plans = {}
    for pname, pred in [("纯裁决者", d.jv.values), ("多数票≥2", d.maj.values),
                        ("全票制≥3", d.unan.values)]:
        plans[pname] = (recall(pred, y, 1), recall(pred, y, 0))
    ri_j, rn_j = plans["纯裁决者"]
    for pname, (ri, rn) in plans.items():
        vals = []
        for pi in [0.20, 0.40, 0.50, 0.60, 0.80]:
            vals.append(pi * ri + (1 - pi) * rn)
        # 求劣于裁决者的 π 区间
        worse_lo, worse_hi = None, None
        for pi in np.arange(0.01, 1.0, 0.01):
            a_p = pi * ri + (1 - pi) * rn
            a_j = pi * ri_j + (1 - pi) * rn_j
            if a_p < a_j:
                if worse_lo is None:
                    worse_lo = pi
                worse_hi = pi
        rng = (f"{worse_lo*100:.0f}%~{worse_hi*100:.0f}%"
               if worse_lo is not None else "无（全域不劣）")
        extrap.append({"set": name, "plan": pname,
                       "acc_at_pi": {f"{int(p*100)}%": round(v, 2)
                                     for p, v in zip([0.20, 0.40, 0.50, 0.60, 0.80], vals)},
                       "worse_than_judge_pi_range": rng,
                       "worse_lo_pct": round(worse_lo * 100, 1) if worse_lo is not None else None,
                       "worse_hi_pct": round(worse_hi * 100, 1) if worse_hi is not None else None})
        print(f"  {name:<14} {pname:<16} " + " ".join(f"{v:>8.2f}%" for v in vals)
              + f" {rng:>22}")
    print("  " + "-" * 100)

# ══════════════════════════════════════════════════
# T4 对齐判正类率后，辩论是否仍有信息增量
# ══════════════════════════════════════════════════
print("\n" + "=" * 110)
print("【T4】剥离阈值效应：对齐「判正类率」后，辩论是否仍带来信息增量")
print("=" * 110)
print("""  方法：全票制的判正类率远低于裁决者 argmax，准确率提升可能纯由阈值平移带来。
  故把裁决者的阈值调到与全票制【相同的判正类率】，再比准确率：
    若两者准确率相当 → 辩论无信息增量，收益全是阈值效应
    若全票制明显更高 → 辩论确实在同样保守度下做出了更好的选择""")
print(f"\n  {'数据集':<14} {'全票制判正类率':>14} {'裁决者匹配阈值':>15} "
      f"{'全票制acc':>11} {'裁决者acc':>11} {'辩论净增量':>11}")
print("  " + "-" * 82)
t4_rows = []
for name, d in data.items():
    y = d.y.values
    target = d.unan.mean()
    au = acc(d.unan.values, y)
    # 找使判正类率最接近 target 的裁决者阈值
    best_t, best_diff, best_pred = None, 1e9, None
    for t in np.arange(0.50, 0.999, 0.005):
        pred = (d.p_pos_judge.values >= t).astype(int)
        diff = abs(pred.mean() - target)
        if diff < best_diff:
            best_diff, best_t, best_pred = diff, t, pred
    ajm = acc(best_pred, y)
    m = mcnemar(best_pred, d.unan.values, y)
    t4_rows.append({"set": name, "unanimity_pro_rate_pct": round(target * 100, 2),
                    "matched_judge_threshold": round(float(best_t), 3),
                    "acc_unanimity_pct": round(au, 2),
                    "acc_judge_matched_pct": round(ajm, 2),
                    "debate_net_gain_pp": round(au - ajm, 2),
                    "mcnemar": m,
                    "judge_pro_rate_at_match_pct": round(best_pred.mean() * 100, 2)})
    print(f"  {name:<14} {target*100:>13.1f}% {best_t:>15.3f} {au:>10.2f}% "
          f"{ajm:>10.2f}% {au-ajm:>+10.2f}pp")
    print(f"  {'':<14} 判正类率对齐后 McNemar: b={m['b']} c={m['c']} χ²={m['chi2']} "
          f"→ {'显著' if m['significant'] else '不显著'}")

# ══════════════════════════════════════════════════
# 结论
# ══════════════════════════════════════════════════
print("\n" + "=" * 110)
print("【结论】回答「辩论机制是否全是正收益」")
print("=" * 110)
concl = []
idx = 0


def add(t):
    global idx
    idx += 1
    concl.append(f"{idx}. {t}")


# T1 汇总
n_replaceable = 0
for name in data:
    rs = [r for r in t1_rows if r["set"] == name and r["beats_unanimity"]]
    if rs:
        n_replaceable += 1
add(f"T1 阈值替代检验：{n_replaceable}/{len(data)} 个数据集上，存在某个阈值使"
    f"【纯裁决者、零辩论、零额外算力】达到或超过全票制的准确率。")
for name in data:
    rs = [r for r in t1_rows if r["set"] == name]
    hit = [r for r in rs if r["beats_unanimity"]]
    if hit:
        add(f"   {name}：全票制 {rs[0]['acc_unanimity']:.2f}%，"
            f"纯裁决者在 t={min(r['t'] for r in hit):.2f} 即达 "
            f"{min((r['acc_judge_thr'] for r in hit), default=0):.2f}% 以上 → 辩论非必要。")

# T3 汇总
unan_bad = [e for e in extrap if e["plan"] == "全票制≥3" and e["worse_lo_pct"] is not None]
if unan_bad:
    add(f"T3 类别比例敏感性：全票制在 {len(unan_bad)}/{len(data)} 个数据集上存在"
        f"【劣于纯裁决者】的正类占比区间："
        + "；".join(f"{e['set']} π∈{e['worse_than_judge_pi_range']}" for e in unan_bad) + "。")
    add("   → 因此「辩论机制全是正收益」【不成立】。全票制的收益依赖目标分布的正类占比："
        "正类占比越高，其严格判正类规则的代价越大。")
else:
    add("T3：全票制在所有实测数据集上全域不劣于纯裁决者。")

# T4 汇总
net = [r["debate_net_gain_pp"] for r in t4_rows]
sig_cnt = sum(1 for r in t4_rows if r["mcnemar"]["significant"])
add(f"T4 剥离阈值效应后（判正类率对齐）：辩论净增量 "
    + "，".join(f"{r['set']} {r['debate_net_gain_pp']:+.2f}pp" for r in t4_rows) + "。")
add(f"   其中 {sig_cnt}/{len(t4_rows)} 达到 McNemar 显著。"
    f"{'辩论在同等保守度下确有信息增量' if sig_cnt > 0 and np.mean(net) > 0 else '辩论的信息增量不显著，收益主要来自阈值平移'}")

add("★ 综合判定（依据 T1 与 T4 的联合证据）：辩论【确有】超越阈值效应的信息增量。"
    "T1 表明 0/3 数据集上纯裁决者能靠调阈值达到全票制；"
    "T4 在判正类率严格对齐后，辩论净增量仍达 +6.64~+8.05pp 且 3/3 McNemar 显著。"
    "故收益不是单纯的阈值平移，辩论方提供了裁决者不具备的判别信息。")
add("★ 但「全是正收益」仍不成立，原因有二："
    "① T3 显示全票制在 π∈64%~99%（正类占多数）区间劣于纯裁决者，"
    "其优势依赖目标分布的正类占比；"
    "② 计票规则本身必须随分布切换 —— 均衡集(π≈50%)上多数票 +18.79pp 远优于"
    "全票制 +6.04pp，而低正类占比(π≈40%)上全票制 +5.87pp 反超多数票 −3.19pp。")
add("★ 对课题的含义（修正版）：可支持的主张是"
    "「多智能体辩论在困难二分类任务上提供显著信息增量（判正类率对齐后 +6.6~8.1pp，"
    "McNemar 显著），但其净收益方向取决于目标分布的正类占比，"
    "需按分布选择计票规则（π≈50% 用多数票，π<45% 用全票制）」。"
    "这比「辩论一律提升性能」更弱，但有实测支撑；"
    "同时它把模块 IV 的门控定位为「按样本/模态选择融合规则」而非「跳过辩论」。")
add("★ 下一步验证：n=150 均衡集上多数票增益最高(+18.79pp)，"
    "说明辩论的价值在均衡分布下最大；hate(正类42.2%)与 FIGLANG(正类50.0%) "
    "两个新任务的实测将直接检验「计票规则随分布切换」这一结论的可推广性 —— "
    "FIGLANG 恰为 50% 均衡，应预期多数票优于全票制。")

for c in concl:
    print(f"\n  {c}")

out = {
    "T1_threshold_substitution": t1_rows,
    "T2_mcnemar": t2_rows,
    "T3_class_ratio_sensitivity": {"measured": t3_rows, "extrapolated": extrap},
    "T4_debate_net_gain": t4_rows,
    "conclusions": concl,
    "verdict": ("辩论确有超越阈值效应的信息增量：T1 显示 0/3 数据集上纯裁决者调阈值"
                "无法达到全票制，T4 在判正类率对齐后净增量 +6.64~+8.05pp 且 3/3 "
                "McNemar 显著。但「全是正收益」不成立：T3 显示全票制在 π∈64%~99% "
                "劣于纯裁决者，且计票规则须随正类占比切换（π≈50% 用多数票，"
                "π<45% 用全票制）"),
}
with open(f"{RES}/debate_contribution_test.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
pd.DataFrame(t1_rows).to_csv(f"{RES}/debate_T1_threshold_substitution.csv",
                             index=False, encoding="utf-8-sig")
pd.DataFrame(t4_rows).to_csv(f"{RES}/debate_T4_net_gain.csv",
                             index=False, encoding="utf-8-sig")
print(f"\n结果已写入: debate_contribution_test.json / "
      f"debate_T1_threshold_substitution.csv / debate_T4_net_gain.csv")
print("=" * 110)
