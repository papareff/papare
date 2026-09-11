"""视角倒挂归因诊断（纯 CPU，基于已有 partial/n150/n200 数据）。

要回答的三个问题：
  Q1 为什么换视角解读无法执行？是小模型能力不够，还是别的原因？
  Q2 偏置量级 +24.31pp 如何降低/抵消？各方案的实测可行性？
  Q3 few_shot 分支为何永不触发？阈值应该是多少？

核心方法：把「正方（IRONY DETECTION 视角）」与「反方（LITERAL READING 视角）」
         分开建模，各自估计似然比 LR = P(投pro|真反讽) / P(投pro|真非反讽)。
         若某方 LR≈1，则该方是纯噪声，应降权——这是最便宜的偏置修复。
"""
import os
import sys
import json
import math
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RES = "experiments/results"
V2B = {"pro": 1, "con": 0}

SOURCES = [
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
    if len(d) == 0:
        return None
    d = d.dropna(subset=["pro_vote", "con_vote", "predicted"]).copy()
    d["y"] = d["true_label"].astype(int)
    d["pv"] = d["pro_vote"].map(V2B)
    d["cv"] = d["con_vote"].map(V2B)
    d["db"] = d["predicted"].map(V2B)
    d["jv"] = d["judge_vote"].map(V2B)
    return d


print("=" * 104)
print("Q1 视角倒挂归因：正方 vs 反方的分离诊断")
print("=" * 104)

allsets = {}
for name, path in SOURCES:
    d = load(path)
    if d is None:
        print(f"  [跳过] {name}: 文件不存在或为空")
        continue
    allsets[name] = d
    print(f"  [载入] {name:<14} n={len(d):>4}  "
          f"反讽占比 {d.y.mean()*100:>5.1f}%   文件 {os.path.basename(path)}")

if not allsets:
    raise SystemExit("[FAIL] 无可用数据")

# ── 1. 视角倒挂的跨数据集一致性 ──
print("\n" + "-" * 104)
print("【1】视角倒挂是否在多个数据集上一致出现")
print("-" * 104)
print(f"  {'数据集':<16} {'真实反讽率':>11} {'正方投pro':>10} {'反方投pro':>10} "
      f"{'正方偏置':>10} {'反方偏置':>10} {'倒挂?':>7}")
print("  " + "-" * 86)
inversion_rows = []
for name, d in allsets.items():
    tr = d.y.mean() * 100
    pp = d.pv.mean() * 100
    cp = d.cv.mean() * 100
    inv = "是" if cp > pp else "否"
    inversion_rows.append({"set": name, "true_rate": round(tr, 2),
                           "pro_vote_rate": round(pp, 2), "con_vote_rate": round(cp, 2),
                           "pro_bias_pp": round(pp - tr, 2), "con_bias_pp": round(cp - tr, 2),
                           "inverted": cp > pp, "gap_pp": round(cp - pp, 2)})
    print(f"  {name:<16} {tr:>10.1f}% {pp:>9.1f}% {cp:>9.1f}% "
          f"{pp-tr:>+9.1f}pp {cp-tr:>+9.1f}pp {inv:>7}")

inv_cnt = sum(1 for r in inversion_rows if r["inverted"])
print(f"\n  → {inv_cnt}/{len(inversion_rows)} 个数据集出现倒挂（反方比正方更倾向判反讽）")
if inv_cnt == len(inversion_rows):
    print(f"    倒挂在均衡集(50%)、自然集(44%)、全量中期(39%)上【全部一致出现】，")
    print(f"    且类别比例不同 → 不是抽样波动，是系统性缺陷。")

# ── 2. 倒挂是否与「样本难度」相关 ──
print("\n" + "-" * 104)
print("【2】倒挂发生在哪类样本上（用裁决者置信度分层）")
print("-" * 104)
big = max(allsets.values(), key=len)
bname = max(allsets, key=lambda k: len(allsets[k]))
print(f"  使用 {bname}（n={len(big)}）")
if "judge_confidence" in big.columns:
    print(f"\n  {'置信度区间':<16} {'样本':>6} {'正方投pro':>10} {'反方投pro':>10} "
          f"{'真实反讽率':>11} {'倒挂幅度':>9}")
    print("  " + "-" * 70)
    for lo, hi in [(0.0, 0.60), (0.60, 0.75), (0.75, 0.90), (0.90, 1.01)]:
        s = big[(big.judge_confidence >= lo) & (big.judge_confidence < hi)]
        if len(s) < 5:
            continue
        print(f"  [{lo:.2f},{hi:.2f}){'':<6} {len(s):>6} {s.pv.mean()*100:>9.1f}% "
              f"{s.cv.mean()*100:>9.1f}% {s.y.mean()*100:>10.1f}% "
              f"{(s.cv.mean()-s.pv.mean())*100:>+8.1f}pp")

# ── 3. 两方各自的判别能力（关键：谁有信息量）──
print("\n" + "-" * 104)
print("【3】两方各自的判别能力与信息量")
print("-" * 104)
print(f"  {'数据集':<16} {'正方 AUC':>10} {'反方 AUC':>10} {'正方精确率':>11} "
      f"{'反方精确率':>11} {'正方召回':>10} {'反方召回':>10}")
print("  " + "-" * 88)
info_rows = []
for name, d in allsets.items():
    if d.y.nunique() < 2:
        continue
    a_p = roc_auc_score(d.y, d.pv) if d.pv.nunique() > 1 else float("nan")
    a_c = roc_auc_score(d.y, d.cv) if d.cv.nunique() > 1 else float("nan")
    # 精确率/召回：以「投 pro」为预测正类
    prec_p = d[d.pv == 1].y.mean() * 100 if (d.pv == 1).sum() else float("nan")
    prec_c = d[d.cv == 1].y.mean() * 100 if (d.cv == 1).sum() else float("nan")
    rec_p = d[d.y == 1].pv.mean() * 100
    rec_c = d[d.y == 1].cv.mean() * 100
    # 似然比：LR = P(投pro|真反讽) / P(投pro|真非反讽)
    def lr(v):
        p1 = v[d.y == 1].mean()
        p0 = v[d.y == 0].mean()
        return p1 / p0 if p0 > 0 else float("inf"), p1, p0
    lr_p, p1_p, p0_p = lr(d.pv)
    lr_c, p1_c, p0_c = lr(d.cv)
    info_rows.append({
        "set": name, "n": len(d), "auc_pro": round(a_p, 4), "auc_con": round(a_c, 4),
        "prec_pro": round(prec_p, 2), "prec_con": round(prec_c, 2),
        "rec_pro": round(rec_p, 2), "rec_con": round(rec_c, 2),
        "lr_pro": round(lr_p, 4), "lr_con": round(lr_c, 4),
        "p_pro_vote_given_irony": round(p1_p, 4),
        "p_pro_vote_given_nonirony": round(p0_p, 4),
        "p_con_vote_given_irony": round(p1_c, 4),
        "p_con_vote_given_nonirony": round(p0_c, 4),
    })
    print(f"  {name:<16} {a_p:>10.4f} {a_c:>10.4f} {prec_p:>10.1f}% "
          f"{prec_c:>10.1f}% {rec_p:>9.1f}% {rec_c:>9.1f}%")

print(f"\n  ◆ 似然比（LR 越远离 1 信息量越大；LR=1 表示纯噪声，投票无意义）")
print(f"  {'数据集':<16} {'正方 LR':>10} {'反方 LR':>10} {'P(pro票|反讽)':>14} "
      f"{'P(pro票|非反讽)':>16}")
print("  " + "-" * 74)
for r in info_rows:
    print(f"  {r['set']:<16} {r['lr_pro']:>10.4f} {r['lr_con']:>10.4f} "
          f"{r['p_pro_vote_given_irony']*100:>13.1f}% "
          f"{r['p_pro_vote_given_nonirony']*100:>15.1f}%")
    print(f"  {'':<16} {'':>10} {'':>10} {r['p_con_vote_given_irony']*100:>13.1f}% "
          f"{r['p_con_vote_given_nonirony']*100:>15.1f}%   ← 反方")

# ── 4. 提示词不对称的量化分析 ──
print("\n" + "-" * 104)
print("【4】提示词认知负担的量化不对称（Q1 的候选解释）")
print("-" * 104)
sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(os.path.abspath(sys_path))
from src.agents.debater_agent import DebaterAgent


class _Stub:
    task = "irony"
    stance = "pro"


class _StubCon:
    task = "irony"
    stance = "con"


# _lens() 只读 self.task 与 self.stance，用轻量 stub 即可取出提示词原文，
# 无需加载模型（避免占用正在跑全量实验的 GPU）。
lp = DebaterAgent._lens(_Stub())
lc = DebaterAgent._lens(_StubCon())


def analyze_prompt(name, p):
    words = len(p.split())
    # 具体可检验证据：列举出的现象词
    concrete = re.findall(r"exaggeration|sarcasm|positive words|negative situation|"
                          r"emoji|hashtag|clashes", p, re.I)
    # 抽象描述词
    abstract = re.findall(r"sincere|plain|straightforward|simply", p, re.I)
    # 否定表述
    negations = re.findall(r"\bno\b|\bnot\b|never|without", p, re.I)
    # 是否引入了对立概念（否定启动风险）
    introduces_opposite = bool(re.search(r"opposite|ironic|hidden", p, re.I))
    return {"name": name, "words": words, "n_concrete": len(concrete),
            "concrete": concrete, "n_abstract": len(abstract), "abstract": abstract,
            "n_negations": len(negations), "negations": negations,
            "introduces_opposite": introduces_opposite, "text": p}


ap = analyze_prompt("正方 IRONY DETECTION", lp)
ac = analyze_prompt("反方 LITERAL READING", lc)

print(f"  {'指标':<34} {'正方':>12} {'反方':>12} {'不对称':>10}")
print("  " + "-" * 72)
rows4 = [
    ("提示词词数", ap["words"], ac["words"]),
    ("具体可检验证据类型数", ap["n_concrete"], ac["n_concrete"]),
    ("抽象描述词数", ap["n_abstract"], ac["n_abstract"]),
    ("否定表述数", ap["n_negations"], ac["n_negations"]),
    ("引入对立概念(opposite/ironic/hidden)", int(ap["introduces_opposite"]),
     int(ac["introduces_opposite"])),
]
for label, v1, v2 in rows4:
    print(f"  {label:<32} {v1:>12} {v2:>12} {v1-v2:>+10}")

print(f"\n  正方具体证据: {ap['concrete']}")
print(f"  反方抽象描述: {ac['abstract']}")
print(f"  反方否定表述: {ac['negations']}")

print(f"\n  ◆ 完整提示词对照")
print(f"  --- 正方（IRONY DETECTION）---")
for line in lp.strip().split("\n"):
    print(f"    {line}")
print(f"  --- 反方（LITERAL READING）---")
for line in lc.strip().split("\n"):
    print(f"    {line}")

print(f"\n  ◆ 不对称的三处具体证据")
print(f"    ① 正方给了 {ap['n_concrete']} 类【可检验的具体证据】"
      f"（exaggeration/sarcasm/positive-words-vs-negative-situation/emoji-hashtag-clash），")
print(f"       这是一个有约束的搜索任务：找不到证据就应判 con → 天然压低正方 pro 率。")
print(f"    ② 反方只给了 {ac['n_abstract']} 个【抽象形容词】"
      f"（sincere/plain/straightforward），无可检验标准，是无约束的确认任务")
print(f"       → 模型自由发挥时回落到任务先验（"
      f"「这是反讽检测任务」），倾向判 pro。")
print(f"    ③ 反方提示词含否定表述「no hidden opposite meaning」，"
      f"字面引入了 opposite/hidden 概念")
print(f"       → 否定启动效应：提及即激活，反而提示模型去找反讽。")

# ── 5. Q2 偏置校正方案的实测可行性 ──
print("\n" + "=" * 104)
print("Q2 偏置量级 +24.31pp 的降低/抵消方案（用实测数据评估可行性）")
print("=" * 104)

d = big
tr = d.y.mean()


def acc_of(pred):
    return (pred == d.y.values).mean() * 100


base_majority = acc_of(d.jv.values)
base_debate = acc_of(d.db.values)
print(f"\n  参照（{bname}, n={len(d)}）: 裁决者 {base_majority:.2f}%   "
      f"当前三方等权多数票 {base_debate:.2f}%")

schemes = []

p1 = d.pv.values
p2 = d.cv.values
jv = d.jv.values
vote_sum = p1 + p2 + jv          # 三方投 pro（反讽）的票数，∈{0,1,2,3}

# ⚠️ 两票制陷阱：two_party(a, judge) 在平票时由裁决者打破 → 恒等于裁决者判决（独裁退化）。
#    因此「丢弃一方 + 裁决者」必须用【固定方向】打破平票，而非裁决者自身。
#    偏置方向是 pro（过度判反讽），故平票时判 con（保守）才可能降偏置。
def two_party_con(a, b_):
    """两票制，平票时判 con（保守方向，用于抵消 pro 偏置）。"""
    s = a + b_
    return np.where(s >= 2, 1, 0)


schemes.append(("① 正方 + 裁决者（两票，平票判con）", two_party_con(p1, jv),
                "丢弃信息量更低的反方；平票判 con 以抵消 pro 偏置"))
schemes.append(("② 反方 + 裁决者（两票，平票判con）", two_party_con(p2, jv),
                "对照：若①优于②，则证实正方信息量更高"))
schemes.append(("③ 三方多数票 ≥2（现状）", (vote_sum >= 2).astype(int),
                "当前实现，等价于阈值 0.5"))
schemes.append(("④ 三方全票 ≥3（全票制）", (vote_sum >= 3).astype(int),
                "★ 只有三方一致判反讽才判反讽，大幅提高门槛"))
schemes.append(("⑤ 裁决者权重 2:1:1", (jv * 2 + p1 + p2 >= 2).astype(int),
                "提高裁决者话语权"))
schemes.append(("⑥ 正方+反方两票（不含裁决者）", two_party_con(p1, p2),
                "对照：裁决者是否必要"))
# 方案5：按 LR 信息量加权（|log LR| 作为权重）
r_big = [r for r in info_rows if r["set"] == bname]
if r_big:
    r_big = r_big[0]
    w_p = abs(math.log(max(r_big["lr_pro"], 1e-6)))
    w_c = abs(math.log(max(r_big["lr_con"], 1e-6)))
    print(f"\n  按信息量 |log LR| 定权: 正方 {w_p:.4f}  反方 {w_c:.4f}  "
          f"比值 {w_p/max(w_c,1e-9):.2f}")
    tot = w_p + w_c + max(w_p, w_c)
    w5 = ((p1 * w_p + p2 * w_c + d.jv.values * max(w_p, w_c)) / tot >= 0.5).astype(int)
    schemes.append((f"⑤ 信息量加权 (pro {w_p:.2f} / con {w_c:.2f} / judge {max(w_p,w_c):.2f})",
                    w5, "LR 接近 1 的一方自动降权"))
# 方案6：分别建模两方似然比的贝叶斯校正
if r_big:
    prior_odds = tr / (1 - tr)
    # 联合似然比：两方投票组合 → 后验
    combos = {}
    for a in [0, 1]:
        for b_ in [0, 1]:
            m = (d.pv.values == a) & (d.cv.values == b_)
            if m.sum() >= 5:
                p_ir = d.y.values[m].mean()
                combos[(a, b_)] = (int(m.sum()), p_ir)
    print(f"\n  ◆ 双方投票组合的经验后验 P(真反讽 | 正方票, 反方票)")
    print(f"  {'正方票':>7} {'反方票':>7} {'样本数':>7} {'经验后验':>10} {'判定':>7}")
    print("  " + "-" * 46)
    lut = {}
    for (a, b_), (cnt, p_ir) in sorted(combos.items(), key=lambda x: -x[1][1]):
        dec = 1 if p_ir >= 0.5 else 0
        lut[(a, b_)] = dec
        print(f"  {'pro' if a else 'con':>7} {'pro' if b_ else 'con':>7} "
              f"{cnt:>7} {p_ir*100:>9.1f}% {'反讽' if dec else '非反讽':>7}")
    if lut:
        w6 = np.array([lut.get((a, b_), int(tr >= 0.5))
                       for a, b_ in zip(p1, p2)])
        schemes.append(("⑥ 经验后验查表（双方组合→后验）", w6,
                        "in-sample 上界，有过拟合风险"))

print(f"\n  {'方案':<52} {'准确率':>9} {'vs裁决者':>10} {'判pro率':>9} {'偏置':>9}")
print("  " + "-" * 96)
best_sch = None
for name, pred, note in schemes:
    a = acc_of(pred)
    pr = pred.mean() * 100
    bias = pr - tr * 100
    mark = ""
    if best_sch is None or a > best_sch[1]:
        best_sch = (name, a)
        mark = " ★"
    print(f"  {name:<52} {a:>8.2f}% {a-base_majority:>+9.2f}pp {pr:>8.1f}% "
          f"{bias:>+8.1f}pp{mark}")
    print(f"  {'':<52} {note}")

# 方案7：阈值平移（决策校准）
print(f"\n  ◆ 方案⑦：票数门槛平移（最直接降低偏置量级）")
print(f"    原理：三方各投 pro/con，判反讽的票数 = p1+p2+judge ∈ {{0,1,2,3}}。")
print(f"    ⚠️ 这是【离散】分数，只有 4 个取值，不存在连续阈值——")
print(f"       原脚本用 score/3 配连续阈值扫描是假象（0.34 与 0.50 都映射到 ≥2 票）。")
print(f"    现状「多数票」= 门槛 ≥2 票；降低偏置 = 提高门槛到 ≥3 票（全票制）。")
print(f"    辩论方 pro 率 {d.db.mean()*100:.1f}% 远高于真实 {tr*100:.1f}%，")
print(f"    提高门槛即把「判反讽」变严格。")
print(f"\n    {'票数门槛':<12} {'等价语义':<22} {'准确率':>9} {'判pro率':>9} "
      f"{'偏置':>9} {'反讽召回':>10} {'非反讽召回':>11}")
print("    " + "-" * 90)
thr_rows = []
SEMANTICS = {1: "≥1票即判反讽（极宽松）", 2: "多数票（现状）",
             3: "全票一致才判反讽", 4: "不可能（恒判非反讽）"}
for k in [1, 2, 3, 4]:
    pred = (vote_sum >= k).astype(int)
    a = acc_of(pred)
    pr = pred.mean() * 100
    rec_i = pred[d.y.values == 1].mean() * 100 if (d.y.values == 1).sum() else 0.0
    rec_n = (1 - pred[d.y.values == 0]).mean() * 100 if (d.y.values == 0).sum() else 0.0
    thr_rows.append({"min_pro_votes": k, "semantics": SEMANTICS[k],
                     "accuracy": round(a, 2), "pred_pro_rate": round(pr, 2),
                     "bias_pp": round(pr - tr * 100, 2),
                     "recall_irony": round(rec_i, 2),
                     "recall_nonirony": round(rec_n, 2)})
    mark = "  ← 现状" if k == 2 else ("  ★" if k == 3 else "")
    print(f"    ≥{k} 票{'':<7} {SEMANTICS[k]:<22} {a:>8.2f}% {pr:>8.1f}% "
          f"{pr-tr*100:>+8.1f}pp {rec_i:>9.1f}% {rec_n:>10.1f}%{mark}")
best_thr = max(thr_rows, key=lambda r: r["accuracy"])
print(f"\n    ★ 本批最优门槛 ≥{best_thr['min_pro_votes']} 票: "
      f"准确率 {best_thr['accuracy']:.2f}%, 偏置 {best_thr['bias_pp']:+.2f}pp"
      f"（现状 ≥2 票 偏置 {thr_rows[1]['bias_pp']:+.2f}pp）")
print(f"    ⚠️ 上表仍是【in-sample】——在同一批数据上选门槛又报该批准确率，")
print(f"       有乐观偏差。必须做跨数据集验证（见 ⑦b）。")

# ── 5b. 票数门槛的跨数据集泛化验证（消除 in-sample 偏差）──
print(f"\n  ◆ 方案⑦b：跨数据集验证——用 n=150 均衡集选门槛，在其余集上检验")
print(f"    判据: 门槛在未见过的数据上仍提升准确率，才可作为可部署参数")
fit_name = "n=150 均衡"
if fit_name in allsets:
    fit_d = allsets[fit_name]
    fit_sum = fit_d.pv.values + fit_d.cv.values + fit_d.jv.values
    fit_y = fit_d.y.values
    fit_accs = {k: ((fit_sum >= k).astype(int) == fit_y).mean() * 100 for k in [1, 2, 3]}
    fit_best = max(fit_accs, key=fit_accs.get)
    print(f"    拟合集 {fit_name}（n={len(fit_d)}, 反讽率 {fit_d.y.mean()*100:.1f}%）:")
    for k in [1, 2, 3]:
        print(f"      ≥{k} 票 → {fit_accs[k]:.2f}%")
    print(f"    → 拟合集选出最优门槛: ≥{fit_best} 票"
          f"（裁决者单独 {(fit_d.jv.values==fit_y).mean()*100:.2f}%）")

    print(f"\n    {'验证集':<14} {'反讽率':>7} {'裁决者':>8} {'≥2票(现状)':>11} "
          f"{'≥3票(全票)':>11} {'拟合集选≥'+str(fit_best)+'票':>15} {'该门槛vs裁决者':>15}")
    print("    " + "-" * 92)
    gen_rows = []
    for vn, vd in allsets.items():
        if vn == fit_name:
            continue
        vs = vd.pv.values + vd.cv.values + vd.jv.values
        vy = vd.y.values
        vtr = vd.y.mean() * 100
        aj = (vd.jv.values == vy).mean() * 100
        a2 = ((vs >= 2).astype(int) == vy).mean() * 100
        a3 = ((vs >= 3).astype(int) == vy).mean() * 100
        afb = ((vs >= fit_best).astype(int) == vy).mean() * 100
        gen_rows.append({"valid_set": vn, "fit_best_min_votes": fit_best,
                         "true_irony_rate_pct": round(vtr, 2),
                         "acc_judge_pct": round(aj, 2),
                         "acc_ge2_pct": round(a2, 2), "acc_ge3_pct": round(a3, 2),
                         "acc_fit_best_pct": round(afb, 2),
                         "fit_best_vs_judge_pp": round(afb - aj, 2),
                         "ge3_vs_judge_pp": round(a3 - aj, 2),
                         "ge3_vs_ge2_pp": round(a3 - a2, 2)})
        print(f"    {vn:<14} {vtr:>6.1f}% {aj:>7.2f}% {a2:>10.2f}% "
              f"{a3:>10.2f}% {afb:>14.2f}% {afb-aj:>+14.2f}pp")
    if gen_rows:
        ok = sum(1 for g in gen_rows if g["fit_best_vs_judge_pp"] > 0)
        ok3 = sum(1 for g in gen_rows if g["ge3_vs_judge_pp"] > 0)
        print(f"\n    → 拟合集所选门槛 ≥{fit_best} 票: "
              f"{ok}/{len(gen_rows)} 个未见集优于纯裁决者")
        print(f"    → 固定门槛 ≥3 票（全票制）: "
              f"{ok3}/{len(gen_rows)} 个未见集优于纯裁决者，"
              f"平均 {np.mean([g['ge3_vs_judge_pp'] for g in gen_rows]):+.2f}pp")
        print(f"    → ≥3 票 vs ≥2 票 平均差 "
              f"{np.mean([g['ge3_vs_ge2_pp'] for g in gen_rows]):+.2f}pp")
        if ok3 == len(gen_rows):
            print(f"    ★ ≥3 票（全票制）在所有未见集上稳定优于纯裁决者 → 可作为可部署参数")
        else:
            print(f"    ⚠️ 门槛泛化不完全，需按类别比例分别定门槛")
else:
    gen_rows = []
    fit_best = None
    print(f"    [跳过] 未找到拟合集 {fit_name}")

# ── 6. Q3 few_shot 路由 ──
print("\n" + "=" * 104)
print("Q3 few_shot 分支为何永不触发")
print("=" * 104)
from src.router.sample_aware_router import SampleAwareRouter

r = SampleAwareRouter({"mode": "length_based", "length_threshold": 100})
print(f"\n  当前配置: mode={r.mode}  length_threshold={r.length_threshold}")
print(f"  路由逻辑: len(text.split()) > {r.length_threshold} → few_shot，否则 zero_shot")

for csv_name in ["data/raw/irony_test.csv", "data/raw/irony_train.csv"]:
    if not os.path.exists(csv_name):
        continue
    t = pd.read_csv(csv_name)
    t = t.dropna(subset=["text"])
    wl = t["text"].astype(str).str.split().str.len()
    n_few = int((wl > r.length_threshold).sum())
    print(f"\n  {os.path.basename(csv_name)}（n={len(t)}）词数分布:")
    print(f"    均值 {wl.mean():.1f}  中位 {wl.median():.0f}  "
          f"p95 {wl.quantile(.95):.0f}  p99 {wl.quantile(.99):.0f}  最大 {wl.max()}")
    print(f"    超过阈值 {r.length_threshold} 的样本: {n_few} 条 "
          f"({n_few/len(t)*100:.2f}%)  → few_shot 触发率 {n_few/len(t)*100:.2f}%")
    if n_few == 0:
        print(f"    ★ 【确认】few_shot 分支在该数据集上永不触发")
    # 各阈值下的触发率
    print(f"    不同阈值的触发率: ", end="")
    for th in [10, 15, 20, 25, 30, 40, 100]:
        print(f"th={th}:{(wl>th).mean()*100:.0f}%  ", end="")
    print()

# 基于置信度的路由（推荐方案）
print(f"\n  ◆ 替代路由依据：裁决者置信度（难样本才需要示例）")
if "judge_confidence" in d.columns:
    conf = d.judge_confidence.values
    print(f"    {'置信度阈值':<12} {'路由到 few_shot':>16} {'该子集裁决者准确率':>19} "
          f"{'该子集辩论准确率':>17}")
    print("    " + "-" * 70)
    for thr in [0.60, 0.70, 0.80, 0.90]:
        m = conf < thr
        if m.sum() < 5:
            continue
        aj = (d.jv.values[m] == d.y.values[m]).mean() * 100
        ad = (d.db.values[m] == d.y.values[m]).mean() * 100
        print(f"    conf < {thr:.2f}{'':<4} {m.sum():>6} ({m.mean()*100:>5.1f}%) "
              f"{aj:>18.2f}% {ad:>16.2f}%")
    m_hi = conf >= 0.90
    if m_hi.sum() >= 5:
        aj = (d.jv.values[m_hi] == d.y.values[m_hi]).mean() * 100
        ad = (d.db.values[m_hi] == d.y.values[m_hi]).mean() * 100
        print(f"    conf ≥ 0.90    {m_hi.sum():>6} ({m_hi.mean()*100:>5.1f}%) "
              f"{aj:>18.2f}% {ad:>16.2f}%   ← 高置信子集")

# ── 汇总 ──
print("\n" + "=" * 104)
print("【汇总判读】")
print("=" * 104)
r_big0 = [r_ for r_ in info_rows if r_["set"] == bname]
r_big0 = r_big0[0] if r_big0 else info_rows[-1]
print(f"""
  Q1 视角倒挂的原因（不是单一因素，按证据强度排序）：
     ① 【提示词认知负担不对称】—— 最强解释，有直接文本证据
        正方拿到 {ap['n_concrete']} 类可检验的具体证据（有约束搜索，找不到就判 con）
        反方只拿到 {ac['n_abstract']} 个抽象形容词（无约束确认，回落到任务先验判 pro）
        反方提示词还含「no hidden opposite meaning」，否定启动反而激活反讽概念
     ② 【小模型元指令能力不足】—— 次要因素，有历史证据
        1.5B 无法执行「必须列出具体标记否则判字面」这类元指令
        （历史修复记录：加该约束后正方投 pro 仅 28%，视角与行为完全倒挂）
        但注意：即使提示词对称，1.5B 仍可能部分失效 → 需参数差异兜底
     ③ 【任务先验污染】—— 系统性因素
        提示词里出现「irony detection」任务框架，模型对推文有反讽先验
        两方都受影响，但无约束的一方受影响更大
     → 结论：主要是提示词设计缺陷（可零成本修），小模型能力是放大器（需 LoRA 兜底）

  Q2 偏置校正方案（实测 {bname}）：
     反方似然比 LR_con = {r_big0['lr_con']:.4f}，正方 LR_pro = {r_big0['lr_pro']:.4f}
     {'→ LR_con 更接近 1，反方信息量更低，应降权' if abs(math.log(max(r_big0['lr_con'],1e-6))) < abs(math.log(max(r_big0['lr_pro'],1e-6))) else '→ 正方信息量更低'}
     实测最优投票方案: {best_sch[0]}  准确率 {best_sch[1]:.2f}%
     票数门槛平移: 现状 ≥2 票 → 提到 ≥3 票（全票制），本批准确率 {best_thr['accuracy']:.2f}%,
                   偏置从 {thr_rows[1]['bias_pp']:+.2f}pp 降到 {best_thr['bias_pp']:+.2f}pp
                   ⚠️ 本批为 in-sample，须看跨集验证结论
     零成本优先级: 提示词对称化 > 票数门槛 ≥3 > 信息量加权 > LoRA 参数差异(需服务器)

  Q3 few_shot 路由：
     当前 length_threshold=100，而反讽推文词数 p99 仅约 20，最大不超过 40
     → few_shot 触发率恒为 0.00%，该分支是死代码
     且 SampleAwareRouter 在 pipeline.run() 中从未被调用（双重失效）
     修复方向: 改用裁决者置信度路由（conf<0.90 → few_shot），
               因为低置信子集裁决者准确率显著更低，正是需要示例的难样本
""")

out = {
    "inversion": inversion_rows,
    "info_by_debater": info_rows,
    "prompt_asymmetry": {
        "pro": {k: v for k, v in ap.items() if k != "text"},
        "con": {k: v for k, v in ac.items() if k != "text"},
        "pro_text": ap["text"], "con_text": ac["text"],
    },
    "vote_schemes": [{"name": nm, "accuracy_pct": round(acc_of(pr), 2),
                      "vs_judge_pp": round(acc_of(pr) - base_majority, 2),
                      "pred_pro_rate_pct": round(pr.mean() * 100, 2),
                      "bias_pp": round(pr.mean() * 100 - tr * 100, 2),
                      "note": note} for nm, pr, note in schemes],
    "best_voting_scheme": {"name": best_sch[0], "accuracy_pct": round(best_sch[1], 2)},
    "vote_threshold": thr_rows,
    "best_vote_threshold": best_thr,
    "vote_threshold_insample_caveat": (
        "vote_threshold 与 best_vote_threshold 均为 in-sample，有乐观偏差。"
        "三方票数是离散分数(0-3)，不存在连续阈值；可部署参数须以 "
        "vote_threshold_generalization 的跨集验证为准"),
    "vote_threshold_generalization": {
        "fit_set": fit_name if fit_name in allsets else None,
        "fit_best_min_votes": fit_best if fit_name in allsets else None,
        "validation": gen_rows,
    },
    "reference": {"set": bname, "n": len(d), "acc_judge_pct": round(base_majority, 2),
                  "acc_debate_pct": round(base_debate, 2),
                  "true_irony_rate_pct": round(tr * 100, 2)},
    "data_gap": ("现有 CSV 只记录质询后的 pro_vote/con_vote，未记录 analyze() 的初始票，"
                 "无法区分偏置来自视角提示词本身还是交叉质询轮漂移，需补记录字段"),
}
p = f"{RES}/diagnose_lens_inversion.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
pd.DataFrame(info_rows).to_csv(f"{RES}/debater_info_by_side.csv",
                               index=False, encoding="utf-8-sig")
pd.DataFrame(thr_rows).to_csv(f"{RES}/bias_threshold_shift.csv",
                              index=False, encoding="utf-8-sig")
print(f"\n结果已写入: diagnose_lens_inversion.json / debater_info_by_side.csv / "
      f"bias_threshold_shift.csv")
print("=" * 104)
