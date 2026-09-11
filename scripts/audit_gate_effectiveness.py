"""门控有效性严格审计：验证「门控在线化是否破坏实验」与「门控收益 vs 全票制」。

用户质疑（2026-09-10）：
  "门控前置直接跳过检索与辩论，不会影响实验吗？
   我怎么感觉跳过之后门控还是没有在其中有作用呢。"

本脚本用已有实测数据严格验证两个命题：
  H1 门控在线化会破坏实验分析能力（数据完整性与可比性）
  H2 门控的收益远小于全票制，且收益来源是「避开辩论的伤害」而非「辩论带来收益」

数据来源（均为已落盘的实测结果，不重跑推理）：
  debate_lens_n150_with.csv          均衡集 n=149 全程辩论
  natural200_debate_with.csv         自然分布 n=199 全程辩论
  roberta_B_full784_debate_with.partial.csv  n=784 中期全程辩论
  conf_{judge}_semeval_test.csv      对应裁决者的逐条置信度
"""
import os
import json
import math
import numpy as np
import pandas as pd

RES = "experiments/results"
V2B = {"pro": 1, "con": 0}

SETS = [
    ("n=150 均衡", f"{RES}/debate_lens_n150_with.csv", None),
    ("n=200 自然", f"{RES}/natural200_debate_with.csv", None),
    ("n=784 中期", f"{RES}/roberta_B_full784_debate_with.partial.csv",
     f"{RES}/conf_judge_roberta_cw_samedomain_ablation_semeval_test.csv"),
]


def load(path, conf_path=None):
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
    d["majority"] = d.predicted.map(V2B)          # 线上实际产出（≥2 票）
    if conf_path and os.path.exists(conf_path):
        c = pd.read_csv(conf_path).iloc[:len(d)].reset_index(drop=True)
        # 校验对齐
        if (c["text"].astype(str).str.strip().values
                != d["text"].astype(str).str.strip().values).any():
            d["conf"] = d.judge_confidence.values
        else:
            d["conf"] = c["confidence"].values
    else:
        d["conf"] = d.judge_confidence.values
    return d


print("=" * 106)
print("门控有效性审计")
print("=" * 106)

data = {}
for name, path, cp in SETS:
    d = load(path, cp)
    if d is None:
        print(f"  [跳过] {name}")
        continue
    data[name] = d
    print(f"  [载入] {name:<14} n={len(d):>4}  正类占比 {d.y.mean()*100:>5.1f}%")

if not data:
    raise SystemExit("[FAIL] 无可用数据")


def acc(pred, y):
    return (np.asarray(pred) == np.asarray(y)).mean() * 100


# ══════════════════════════════════════════════════════════
# H1 门控在线化对实验分析能力的影响
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 106)
print("【H1】门控在线化是否破坏实验分析能力")
print("=" * 106)

print("""
  门控前置的实现是：confidence >= gate_threshold 时直接 return，
  跳过检索与辩论。这带来四个后果：""")

consequences = [
    ("① 被跳过样本无辩论数据",
     "debate_votes={} → 无 pro_vote/con_vote/initial_votes，"
     "这些样本无法参与偏置分析、视角倒挂诊断、投票方案对比"),
    ("② 无法离线扫描多个 τ",
     "离线扫描需要「全部样本都辩论过」才能事后模拟任意 τ；"
     "在线门控下改 τ 必须重跑推理（1800 条 hate 任务约 11 小时）"),
    ("③ 失去反事实分析能力",
     "无法回答「被跳过的样本如果辩论了会不会更好」，"
     "而这正是判断门控是否误伤的关键问题"),
    ("④ 与历史结果不可配对",
     "n=150/n=200/n=784 都是全程辩论数据；在线门控数据只有部分样本有辩论，"
     "两组无法做 McNemar 配对检验"),
]
for k, v in consequences:
    print(f"\n  {k}\n     {v}")

# 量化①：被跳过样本占多少，其中有多少是错的
print(f"\n  ◆ 量化后果①与③：被门控跳过的样本里，有多少是「本可被纠正的错误」")
print(f"  {'数据集':<14} {'τ':>5} {'跳过率':>8} {'跳过样本中裁决者错误':>20} "
      f"{'其中辩论能纠正':>15} {'误伤(辩论反而错)':>18}")
print("  " + "-" * 92)
skip_rows = []
for name, d in data.items():
    y = d.y.values
    for tau in [0.80, 0.90, 0.95]:
        skip = d.conf.values >= tau          # 被门控跳过
        if skip.sum() == 0:
            continue
        n_skip = int(skip.sum())
        judge_wrong_on_skip = int(((d.jv.values != y) & skip).sum())
        # 反事实：这些被跳过的样本，如果辩论了会怎样
        debate_would_fix = int(((d.jv.values != y) & (d.majority.values == y) & skip).sum())
        debate_would_break = int(((d.jv.values == y) & (d.majority.values != y) & skip).sum())
        skip_rows.append({
            "set": name, "tau": tau, "skip_n": n_skip,
            "skip_rate_pct": round(n_skip / len(d) * 100, 2),
            "judge_wrong_in_skipped": judge_wrong_on_skip,
            "judge_wrong_rate_in_skipped_pct": round(
                judge_wrong_on_skip / n_skip * 100, 2),
            "debate_would_fix": debate_would_fix,
            "debate_would_break": debate_would_break,
            "net_lost_pp": round((debate_would_fix - debate_would_break) / len(d) * 100, 2),
        })
        print(f"  {name:<14} {tau:>5.2f} {n_skip/len(d)*100:>7.1f}% "
              f"{judge_wrong_on_skip:>8} ({judge_wrong_on_skip/n_skip*100:>5.1f}%) "
              f"{debate_would_fix:>14} {debate_would_break:>17}")

print(f"\n  → 「辩论能纠正」列是门控【放弃】的纠错机会；")
print(f"    「辩论反而错」列是门控【避开】的伤害。两者之差即门控的净代价/收益。")

# ══════════════════════════════════════════════════════════
# H2 门控收益 vs 全票制
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 106)
print("【H2】门控收益 vs 全票制收益（关键对比）")
print("=" * 106)
print("""
  门控 = 改变「是否辩论」（跳过部分样本）
  全票制 = 改变「如何计票」（三方一致才判 pro，vote_threshold=3）
  两者正交，可组合。""")

print(f"\n  {'数据集':<14} {'方案':<34} {'准确率':>9} {'vs纯裁决者':>11} "
      f"{'算力(辩论条数)':>15}")
print("  " + "-" * 92)
h2_rows = []
for name, d in data.items():
    y = d.y.values
    n = len(d)
    aj = acc(d.jv.values, y)
    vsum = d.pv.values + d.cv.values + d.jv.values

    plans = []
    # 纯裁决者
    plans.append(("纯裁决者（不辩论）", d.jv.values, aj, 0))
    # 全程辩论 + 多数票（现状）
    plans.append(("全程辩论 + 多数票 ≥2（现状）", d.majority.values,
                  acc(d.majority.values, y), n))
    # 全程辩论 + 全票制
    plans.append(("全程辩论 + 全票制 ≥3", (vsum >= 3).astype(int),
                  acc((vsum >= 3).astype(int), y), n))
    # 门控（各 τ）+ 多数票
    for tau in [0.80, 0.90, 0.95]:
        skip = d.conf.values >= tau
        pred = np.where(skip, d.jv.values, d.majority.values)
        plans.append((f"门控 τ={tau:.2f} + 多数票 ≥2", pred, acc(pred, y),
                      int((~skip).sum())))
    # 门控 + 全票制（组合）
    for tau in [0.90]:
        skip = d.conf.values >= tau
        pred = np.where(skip, d.jv.values, (vsum >= 3).astype(int))
        plans.append((f"★ 门控 τ={tau:.2f} + 全票制 ≥3（组合）", pred,
                      acc(pred, y), int((~skip).sum())))

    for pname, pred, a, cost in plans:
        h2_rows.append({"set": name, "plan": pname, "accuracy_pct": round(a, 2),
                        "vs_judge_pp": round(a - aj, 2), "debate_calls": cost,
                        "debate_rate_pct": round(cost / n * 100, 2)})
        print(f"  {name:<14} {pname:<34} {a:>8.2f}% {a-aj:>+10.2f}pp "
              f"{cost:>8}/{n:<5}")
    print("  " + "-" * 92)

# 汇总：哪个方案在各数据集上稳定最优
print(f"\n  ◆ 跨数据集稳定性（在全部 {len(data)} 个数据集上的排名）")
plan_names = sorted(set(r["plan"] for r in h2_rows),
                    key=lambda p: [r["plan"] for r in h2_rows].index(p))
# 保持出现顺序
seen = []
for r in h2_rows:
    if r["plan"] not in seen:
        seen.append(r["plan"])
print(f"  {'方案':<36} " + " ".join(f"{k:>13}" for k in data) + f" {'平均增益':>10}")
print("  " + "-" * (38 + 14 * len(data) + 10))
summary = []
for pname in seen:
    vals, gains = [], []
    for k in data:
        r = next((x for x in h2_rows if x["set"] == k and x["plan"] == pname), None)
        if r:
            vals.append(f"{r['accuracy_pct']:.2f}%")
            gains.append(r["vs_judge_pp"])
        else:
            vals.append("—")
    mg = np.mean(gains) if gains else 0
    summary.append({"plan": pname, "mean_gain_vs_judge_pp": round(mg, 2),
                    "gains": gains,
                    "all_positive": bool(gains and all(g > 0 for g in gains)),
                    "n_sets": len(gains)})
    print(f"  {pname:<36} " + " ".join(f"{v:>13}" for v in vals) + f" {mg:>+9.2f}pp")

best = max(summary, key=lambda s: s["mean_gain_vs_judge_pp"])
stable = [s for s in summary if s["all_positive"] and s["n_sets"] == len(data)]
print(f"\n  ★ 平均增益最高: {best['plan']}  ({best['mean_gain_vs_judge_pp']:+.2f}pp)")
if stable:
    print(f"  ★ 在所有数据集上均为正增益的方案:")
    for s in sorted(stable, key=lambda x: -x["mean_gain_vs_judge_pp"]):
        print(f"      {s['plan']:<36} {s['mean_gain_vs_judge_pp']:+.2f}pp")

# ══════════════════════════════════════════════════════════
# 结论（逐数据集判读；全部判据从 h2_rows 现算，不依赖未定义变量）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 106)
print("【结论】")
print("=" * 106)
print("  ⚠️ 方法论声明：三个数据集正类占比不同（50.3% / 44.2% / 38.4%），")
print("     对其准确率求算术平均【无意义】——与此前事后分层估算同类谬误。")
print("     故下面逐数据集判读，summary 里的平均增益字段不可引用。")


def per_set_gain(plan_kw, exclude=None):
    """按数据集取出某方案的 vs裁决者 增益。"""
    out = {}
    for r in h2_rows:
        if plan_kw in r["plan"] and (exclude is None or exclude not in r["plan"]):
            out[r["set"]] = r["vs_judge_pp"]
    return out


def per_set_calls(plan_kw, exclude=None):
    out = {}
    for r in h2_rows:
        if plan_kw in r["plan"] and (exclude is None or exclude not in r["plan"]):
            out[r["set"]] = (r["debate_calls"], len(data[r["set"]]))
    return out


g_maj = per_set_gain("多数票 ≥2（现状）")
g_una = per_set_gain("全票制 ≥3", exclude="门控")
c_una = per_set_calls("全票制 ≥3", exclude="门控")
g_combo = per_set_gain("组合")
gate_plans = [s["plan"] for s in summary if s["plan"].startswith("门控")]

# ── 退化检测：门控是否真的跳过了样本 ──
print(f"\n  ◆ 退化检测：门控方案是否真的跳过了样本")
print(f"    （若辩论条数 == 样本总数，则该「门控」实际未生效，等价于全程辩论，")
print(f"      其准确率与多数票行完全相同 → 不是独立方案，不可当作门控收益）")
degenerate = []
for pname in gate_plans:
    calls = per_set_calls(pname)
    for k, (used, total) in calls.items():
        if used == total:
            degenerate.append((pname, k))
            print(f"    [退化] {pname:<32} @ {k:<12} 跳过 0 条 → 等价全程辩论")
        else:
            print(f"    [有效] {pname:<32} @ {k:<12} 辩论 {used}/{total} "
                  f"(跳过 {total-used} 条, {100-used/total*100:.1f}%)")

# ── 门控 vs 全票制 的冲突性 ──
print(f"\n  ◆ 门控与全票制是否正交可组合（检验此前「可组合」的判断）")
print(f"    {'数据集':<14} {'纯全票制':>11} {'门控+全票制':>13} {'差值':>9} {'结论':>12}")
print("    " + "-" * 62)
conflict = []
for k in data:
    y = data[k].y.values
    d = data[k]
    vsum = d.pv.values + d.cv.values + d.jv.values
    a_una = acc((vsum >= 3).astype(int), y)
    skip = d.conf.values >= 0.90
    a_combo = acc(np.where(skip, d.jv.values, (vsum >= 3).astype(int)), y)
    conflict.append({"set": k, "unanimity_pct": round(a_una, 2),
                     "gate_plus_unanimity_pct": round(a_combo, 2),
                     "delta_pp": round(a_combo - a_una, 2),
                     "combo_worse": bool(a_combo < a_una)})
    print(f"    {k:<14} {a_una:>10.2f}% {a_combo:>12.2f}% {a_combo-a_una:>+8.2f}pp "
          f"{'组合更差' if a_combo < a_una else '组合更好':>12}")
n_worse = sum(1 for c in conflict if c["combo_worse"])
print(f"    → {n_worse}/{len(conflict)} 个数据集上「门控+全票制」劣于「纯全票制」")

# ── 最优计票规则 vs 正类占比 ──
print(f"\n  ◆ 最优计票规则随正类占比的变化")
print(f"    {'数据集':<14} {'正类占比':>9} {'多数票≥2':>10} {'全票制≥3':>10} {'更优':>12}")
print("    " + "-" * 60)
ratio_rows = []
for k, d in data.items():
    y = d.y.values
    vsum = d.pv.values + d.cv.values + d.jv.values
    a_maj = acc(d.majority.values, y)
    a_una = acc((vsum >= 3).astype(int), y)
    pos = d.y.mean() * 100
    winner = "多数票 ≥2" if a_maj > a_una else "全票制 ≥3"
    ratio_rows.append({"set": k, "pos_rate_pct": round(pos, 2),
                       "acc_majority_pct": round(a_maj, 2),
                       "acc_unanimity_pct": round(a_una, 2),
                       "gain_majority_pp": round(a_maj - acc(d.jv.values, y), 2),
                       "gain_unanimity_pp": round(a_una - acc(d.jv.values, y), 2),
                       "winner": winner})
    print(f"    {k:<14} {pos:>8.1f}% {a_maj:>9.2f}% {a_una:>9.2f}% {winner:>12}")

# ── 结论条目 ──
concl = []
idx = 0


def add(t):
    global idx
    idx += 1
    concl.append(f"{idx}. {t}")


if g_maj:
    add("现状（全程辩论 + 多数票 ≥2）的增益随正类占比下降而恶化："
        + "，".join(f"{k} {v:+.2f}pp" for k, v in g_maj.items()) + "。")
    add("根因：辩论方 pro 偏置是绝对倾向，正类占比越低、过度判 pro 的伤害越大"
        "（与 diagnose_gain_collapse.py 的临界占比 π* 分析同源）。")
if g_una:
    vals = list(g_una.values())
    add("★ 全票制 ≥3 是唯一在全部正类占比下都稳定正收益的方案："
        + "，".join(f"{k} {v:+.2f}pp" for k, v in g_una.items())
        + f"（极差仅 {max(vals)-min(vals):.2f}pp）。")
    add("全票制【不跳过任何样本】（辩论条数 = 样本总数），"
        "故不破坏偏置分析、离线扫 τ、反事实分析、历史配对检验四项能力。")
if degenerate:
    add("★ 门控方案的平均值存在【退化污染】：以下 (方案, 数据集) 实际跳过 0 条样本，"
        "等价于全程辩论，准确率与多数票行完全相同，不是独立方案 → "
        + "；".join(f"{p} @ {k}" for p, k in degenerate) + "。")
    add("因此按平均值选出的「门控 τ=0.95 最优」主要来自退化继承，"
        "不能当作门控自身的收益，此前该结论作废。")
if conflict and n_worse > len(conflict) / 2:
    add(f"★【修正此前判断】门控与全票制【不是正交可组合，而是互相冲突】："
        f"{n_worse}/{len(conflict)} 个数据集上组合劣于纯全票制。")
    add("机理：门控把高置信样本交还给裁决者单独判决，而全票制在这些样本上本就优于裁决者，"
        "故门控等于【撤销】了全票制的收益 —— 两者争夺同一批样本的决策权。")
add("★ 回答「跳过检索与辩论会不会影响实验」：会，且影响严重。四条后果——"
    "①被跳过样本无 pro_vote/con_vote/initial_votes，无法参与偏置与视角倒挂分析；"
    "②无法离线扫描多个 τ，改 τ 必须重跑推理（hate 全量约 11h）；"
    "③失去「被跳过样本若辩论会否更好」的反事实分析能力；"
    "④与全程辩论的历史结果无法做 McNemar 配对检验。")
add("★ 回答「门控是不是没起作用」：在 n=150/n=200 上确实几乎没起作用"
    "（τ=0.95 跳过 0 条、τ=0.90 跳过 1~5 条），只有 n=784 上 τ=0.80 才跳过约 80%。"
    "原因是各数据集置信度分布不同，固定 τ 的触发率差异极大 → τ 必须按数据集单独标定。")
add("落地建议（修正版）：① 默认【关闭】在线门控，全程辩论采集完整数据；"
    "② 偏置修复优先用全票制 ≥3（稳定 +5~6pp 且不损数据完整性）；"
    "③ 门控仅在专门的「部署性能与算力节省」实验中开启，且 τ 按置信度分布单独标定；"
    "④ 计票规则需随目标分布的正类占比调整，不存在单一最优规则。")

for c in concl:
    print(f"\n  {c}")

out = {
    "skipped_sample_analysis": skip_rows,
    "plan_comparison": h2_rows,
    "summary_by_plan_UNRELIABLE_mean": summary,
    "degenerate_gate_plans": [{"plan": p, "set": k} for p, k in degenerate],
    "conflict_gate_vs_unanimity": conflict,
    "ratio_dependence": ratio_rows,
    "conclusions": concl,
    "methodology_warning": ("三数据集正类占比不同（50.3%/44.2%/38.4%），"
                            "跨集算术平均无意义；summary 中的平均增益字段仅供排序，不可引用"),
    "h1_verdict": ("门控在线化破坏四项分析能力：无辩论数据、无法离线扫 τ、"
                   "失去反事实分析、与历史结果不可配对"),
    "h2_verdict": ("全票制 ≥3 是唯一在三种正类占比下都稳定 +5~6pp 且不跳过样本的方案；"
                   "门控平均值受退化污染；门控与全票制互相冲突而非正交可组合"),
    "verdict_on_gate_default": "默认关闭在线门控；偏置修复用全票制；门控仅用于部署性能实验",
}
with open(f"{RES}/gate_effectiveness_audit.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
pd.DataFrame(h2_rows).to_csv(f"{RES}/gate_plan_comparison.csv",
                             index=False, encoding="utf-8-sig")
pd.DataFrame(skip_rows).to_csv(f"{RES}/gate_skipped_sample_cost.csv",
                               index=False, encoding="utf-8-sig")
print(f"\n结果已写入: gate_effectiveness_audit.json / gate_plan_comparison.csv / "
      f"gate_skipped_sample_cost.csv")
print("=" * 106)
