"""辩论前廉价代理信号：预测「裁决者会出错」，替代需要辩论方先出票的分歧信号。

背景（gate_calibration_and_trigger.py 结论）：
  最强门控信号是「裁决者≠辩论方pro票」，AUC=0.7366，
  但它需要辩论方先出票 → 要判断该不该辩论必须先辩论，门控省算力的前提被破坏。

  已排除的伪方案：
    • top-2 logit 差 —— 二分类下 = 2×p_max−1，是 max 置信度的单调变换，AUC 必然相同，零信息增量
    • 反讽概率 p_irony —— AUC 仅 0.5386，接近随机

本脚本尝试真正「辩论前可得」的信号：文本表层特征（零 LLM 调用，CPU 毫秒级）
  反讽的语言学线索：情感极性冲突、标点密度、话题标签、全大写、长度异常、
  积极词+消极语境并置、表情符号、引号（回声式提及 echoic mention）

方法：在 n=150 上用逻辑回归/随机森林学习「裁决者出错」的可预测性，
     5 折交叉验证给出无偏 AUC（in-sample 会严重过拟合，不可信）。
"""
import os
import json
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve

RES = "experiments/results"
DEBATE = f"{RES}/debate_lens_n150_with.csv"
BASE = f"{RES}/debate_lens_n150_judge_only.csv"
NATURAL = f"{RES}/natural200_debate_with.csv"

# ── 情感词典（用于极性冲突检测，反讽的核心线索）──
POS_WORDS = set("""good great awesome amazing love lovely wonderful fantastic excellent
best perfect beautiful happy glad nice sweet brilliant superb fantastic terrific
enjoy enjoying enjoyed thanks thank grateful blessed lucky yay woohoo hooray
adorable cute fun funny hilarious lol xoxo congratulations congrats
""".split())
NEG_WORDS = set("""bad terrible awful horrible hate worst ugly sad angry annoying
stupid dumb fail failed failure sucks boring bored waste useless pathetic
disgusting horrible nightmare worst awful poor broken dead death kill killed
crash crisis disaster tragic sorry unfortunately sadly regret disappoint
""".split())
INTENSIFIERS = set("""so very really totally completely absolutely utterly incredibly
extremely super mega ultra totally literally actually definitely certainly
""".split())


def extract_features(text):
    """从文本抽取辩论前可得的表层特征（零模型调用）。"""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    words = re.findall(r"[a-z']+", t)
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)

    ws = set(words)
    n_pos = len(ws & POS_WORDS)
    n_neg = len(ws & NEG_WORDS)
    n_int = len(ws & INTENSIFIERS)

    # 极性冲突：反讽的典型标志（积极词+消极词并置）
    polarity_conflict = int(n_pos > 0 and n_neg > 0)
    polarity_ratio = (n_pos - n_neg) / n_words

    # 标点与语气线索
    n_excl = text.count("!")
    n_ques = text.count("?")
    n_ellipsis = len(re.findall(r"\.{2,}|…", text))
    n_quote = text.count('"') + text.count("'") + text.count("“") + text.count("”")

    # 话题标签与提及
    n_hashtag = len(re.findall(r"#\w+", text))
    n_mention = len(re.findall(r"@\w+", text))
    hashtag_words = set(w.strip("#").lower() for w in re.findall(r"#\w+", text))
    hashtag_polarity = int(
        bool(hashtag_words & POS_WORDS) or bool(hashtag_words & NEG_WORDS))
    # 标签与正文极性冲突（如正文消极 + #blessed）
    tag_text_conflict = int(
        hashtag_polarity and polarity_ratio != 0 and
        ((n_pos > 0 and (hashtag_words & NEG_WORDS)) or
         (n_neg > 0 and (hashtag_words & POS_WORDS))))

    # 全大写比例（ shouting，反讽常见）
    upper_ratio = sum(1 for c in text if c.isupper()) / n_chars
    long_upper = int(bool(re.search(r"\b[A-Z]{4,}\b", text)))

    # 表情符号
    n_emoji = len(re.findall(r"[:;]-?[\)\(DPpOo/\|]|<3|\bxd\b|\blol\b|\blmao\b", t))

    # 长度异常（过长或过短）
    len_z = (n_words - 13.7) / 8.0   # SemEval 均值 13.7

    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "avg_word_len": n_chars / n_words,
        "n_pos": n_pos, "n_neg": n_neg, "n_intensifier": n_int,
        "polarity_conflict": polarity_conflict,
        "polarity_ratio": polarity_ratio,
        "pos_neg_sum": n_pos + n_neg,
        "n_excl": n_excl, "n_ques": n_ques,
        "excl_ques_ratio": n_excl / max(n_ques, 1),
        "n_ellipsis": n_ellipsis, "n_quote": n_quote,
        "n_hashtag": n_hashtag, "n_mention": n_mention,
        "hashtag_polarity": hashtag_polarity,
        "tag_text_conflict": tag_text_conflict,
        "upper_ratio": upper_ratio, "long_upper": long_upper,
        "n_emoji": n_emoji, "len_z": len_z,
    }


FEAT_NAMES = list(extract_features("test").keys())


def build_xy(texts, judge_wrong, judge_conf=None):
    rows = [extract_features(t) for t in texts]
    X = pd.DataFrame(rows)[FEAT_NAMES].values.astype(float)
    if judge_conf is not None:
        X = np.hstack([X, np.asarray(judge_conf).reshape(-1, 1)])
        names = FEAT_NAMES + ["judge_confidence"]
    else:
        names = FEAT_NAMES
    return X, np.asarray(judge_wrong).astype(int), names


def cv_auc(X, y, model, n_splits=5, seed=42):
    """5 折交叉验证 AUC（无偏估计，in-sample 会严重过拟合）。"""
    if len(np.unique(y)) < 2:
        return float("nan"), None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    return roc_auc_score(y, proba), proba


print("=" * 100)
print("辩论前廉价代理信号：预测「裁决者会出错」")
print("=" * 100)

df = pd.read_csv(DEBATE)
df["judge_b"] = df["judge_vote"].map({"pro": 1, "con": 0})
df["gold_b"] = df["true_label"].astype(int)
df["judge_wrong"] = (df["judge_b"] != df["gold_b"]).astype(int)

print(f"样本 n={len(df)}   裁决者出错率 = {df.judge_wrong.mean()*100:.2f}%")
print(f"特征数 = {len(FEAT_NAMES)}（文本表层，零 LLM 调用）")

# ── 参照：已有信号的 AUC ──
print("\n" + "─" * 100)
print("【参照】辩论前 / 辩论后信号的 AUC（目标：预测裁决者出错）")
print("─" * 100)
refs = [
    ("max 置信度（辩论前，已有）", -df.judge_confidence.values, "辩论前"),
    ("反讽概率 p_irony（辩论前，已有）",
     np.where(df.judge_b == 1, df.judge_confidence, 1 - df.judge_confidence), "辩论前"),
    ("辩论方分歧（辩论后，需先辩论）",
     (df.judge_b != df.pro_vote.map({"pro": 1, "con": 0})).astype(int).values, "辩论后"),
]
for name, sig, stage in refs:
    try:
        a = roc_auc_score(df.judge_wrong, sig)
        print(f"  {name:<36} AUC={a:.4f}   [{stage}]")
    except ValueError:
        print(f"  {name:<36} 无法计算   [{stage}]")
print("  → 目标：找到 AUC 显著高于置信度 0.6045 的**辩论前**信号")

# ── 1. 单特征可预测性筛查 ──
print("\n" + "─" * 100)
print("【1】单特征 AUC 筛查（哪些表层特征单独就有预测力）")
print("─" * 100)
Xf, y, names = build_xy(df.text.values, df.judge_wrong.values)
single = []
for i, nm in enumerate(FEAT_NAMES):
    col = Xf[:, i]
    if np.std(col) < 1e-9:
        continue
    try:
        a = roc_auc_score(y, col)
        single.append((nm, a, abs(a - 0.5)))
    except ValueError:
        pass
single.sort(key=lambda x: -x[2])
print(f"  {'特征':<24} {'AUC':>8} {'偏离随机':>10}")
print("  " + "-" * 46)
for nm, a, d in single[:12]:
    flag = "★" if d > 0.10 else ""
    print(f"  {nm:<24} {a:>8.4f} {d:>+9.4f}  {flag}")

# ── 2. 多特征模型（交叉验证）──
print("\n" + "─" * 100)
print("【2】多特征模型 5 折交叉验证 AUC（无偏估计）")
print("─" * 100)
MODELS = {
    "逻辑回归(仅文本特征)": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
    "逻辑回归(文本+置信度)": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
    "随机森林(仅文本特征)": RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42),
    "随机森林(文本+置信度)": RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42),
}
res_models = {}
print(f"  {'模型':<28} {'CV AUC':>9} {'in-sample AUC':>15} {'过拟合差':>10}")
print("  " + "-" * 66)
for name, mdl in MODELS.items():
    use_conf = "置信度" in name
    X, yy, _ = build_xy(df.text.values, df.judge_wrong.values,
                        df.judge_confidence.values if use_conf else None)
    auc_cv, proba = cv_auc(X, yy, mdl)
    mdl.fit(X, yy)
    auc_in = roc_auc_score(yy, mdl.predict_proba(X)[:, 1])
    res_models[name] = {"cv_auc": auc_cv, "in_sample_auc": auc_in,
                        "overfit_gap": auc_in - auc_cv}
    print(f"  {name:<28} {auc_cv:>9.4f} {auc_in:>15.4f} {auc_in-auc_cv:>+10.4f}")

best_name = max(res_models, key=lambda k: (res_models[k]["cv_auc"]
                                           if not np.isnan(res_models[k]["cv_auc"]) else -1))
best_cv = res_models[best_name]["cv_auc"]
print(f"\n  ★ 最优: {best_name}  CV AUC={best_cv:.4f}")
print(f"    vs 置信度基线 0.6045: {best_cv-0.6045:+.4f}")
print(f"    vs 辩论后分歧信号 0.7366: {best_cv-0.7366:+.4f}")
if best_cv > 0.6045:
    print("    → 辩论前代理信号优于置信度，门控可在辩论前决策，算力节省前提成立")
else:
    print("    → 辩论前代理信号仍不如置信度，门控的算力节省前提难以成立")

# ── 3. 特征重要性 ──
print("\n" + "─" * 100)
print("【3】特征重要性（随机森林，文本+置信度）")
print("─" * 100)
X, yy, names = build_xy(df.text.values, df.judge_wrong.values, df.judge_confidence.values)
rf = RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=5,
                            random_state=42).fit(X, yy)
imp = sorted(zip(names, rf.feature_importances_), key=lambda x: -x[1])
print(f"  {'特征':<26} {'重要性':>9}")
print("  " + "-" * 38)
for nm, v in imp[:12]:
    print(f"  {nm:<26} {v:>9.4f}")

# ── 4. 用代理信号重构门控（含效率约束）──
print("\n" + "─" * 100)
print("【4】用辩论前代理信号重构门控（5 折 out-of-fold 预测，无信息泄漏）")
print("─" * 100)
X, yy, _ = build_xy(df.text.values, df.judge_wrong.values, df.judge_confidence.values)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
oof = cross_val_predict(mdl, X, yy, cv=skf, method="predict_proba")[:, 1]

df["proxy_risk"] = oof          # 裁决者出错的风险（辩论前可得）
base = pd.read_csv(BASE)
df["judge_correct_b"] = 1 - df.judge_wrong
merged = df.merge(base[["sample_id", "correct"]].rename(columns={"correct": "jc"}),
                  on="sample_id", how="left")
if "jc" not in merged or merged.jc.isna().all():
    merged["jc"] = df.judge_correct_b
acc_j = merged.jc.mean() * 100
acc_d = merged.correct.mean() * 100

print(f"  参照: 纯裁决者 {acc_j:.2f}%   全程辩论 {acc_d:.2f}%   "
      f"辩论增益 {acc_d-acc_j:+.2f}pp")
print(f"\n  {'触发预算':<10} {'阈值':>8} {'实际触发率':>11} {'准确率':>9} "
      f"{'vs纯裁决者':>11} {'vs全程辩论':>11}")
print("  " + "-" * 66)
proxy_rows = []
for budget in [0.20, 0.30, 0.40, 0.50, 0.60]:
    thr = np.quantile(merged.proxy_risk, 1 - budget)
    trig = merged.proxy_risk >= thr
    pred = np.where(trig, merged.predicted.map({"pro": 1, "con": 0}), merged.judge_b)
    a = (pred == merged.gold_b).mean() * 100
    proxy_rows.append({"budget_pct": budget * 100, "threshold": round(float(thr), 4),
                       "actual_trigger_pct": round(trig.mean() * 100, 2),
                       "accuracy_pct": round(a, 2),
                       "vs_judge_pp": round(a - acc_j, 2),
                       "vs_debate_pp": round(a - acc_d, 2)})
    print(f"  ≤{budget*100:.0f}%{'':<6} {thr:>8.4f} {trig.mean()*100:>10.1f}% "
          f"{a:>8.2f}% {a-acc_j:>+10.2f}pp {a-acc_d:>+10.2f}pp")

# 与置信度门控在相同预算下对比
print(f"\n  同预算下对比（代理信号 vs 原始置信度门控）:")
print(f"  {'触发预算':<10} {'代理信号准确率':>15} {'置信度门控准确率':>17} {'差值':>9}")
print("  " + "-" * 55)
cmp_rows = []
for budget in [0.20, 0.30, 0.40, 0.50, 0.60]:
    thr = np.quantile(merged.proxy_risk, 1 - budget)
    trig_p = merged.proxy_risk >= thr
    pred_p = np.where(trig_p, merged.predicted.map({"pro": 1, "con": 0}), merged.judge_b)
    acc_p = (pred_p == merged.gold_b).mean() * 100

    thr_c = np.quantile(merged.judge_confidence, budget)
    trig_c = merged.judge_confidence < thr_c
    pred_c = np.where(trig_c, merged.predicted.map({"pro": 1, "con": 0}), merged.judge_b)
    acc_c = (pred_c == merged.gold_b).mean() * 100
    cmp_rows.append({"budget_pct": budget * 100, "proxy_acc": round(acc_p, 2),
                     "conf_acc": round(acc_c, 2), "delta_pp": round(acc_p - acc_c, 2)})
    print(f"  ≤{budget*100:.0f}%{'':<6} {acc_p:>14.2f}% {acc_c:>16.2f}% "
          f"{acc_p-acc_c:>+8.2f}pp")

# ── 5. 在 n=200 自然分布上验证泛化（若已完成）──
print("\n" + "─" * 100)
print("【5】跨数据集泛化验证（n=150 训练 → n=200 自然分布测试）")
print("─" * 100)
if os.path.exists(NATURAL):
    nat = pd.read_csv(NATURAL)
    nat["judge_b"] = nat["judge_vote"].map({"pro": 1, "con": 0})
    nat["gold_b"] = nat["true_label"].astype(int)
    nat["judge_wrong"] = (nat["judge_b"] != nat["gold_b"]).astype(int)
    Xn, yn, _ = build_xy(nat.text.values, nat.judge_wrong.values,
                         nat.judge_confidence.values)
    mdl_final = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=0.5)).fit(X, yy)
    pn = mdl_final.predict_proba(Xn)[:, 1]
    try:
        auc_nat = roc_auc_score(yn, pn)
        auc_nat_conf = roc_auc_score(yn, -nat.judge_confidence.values)
        auc_nat_dis = roc_auc_score(
            yn, (nat.judge_b != nat.pro_vote.map({"pro": 1, "con": 0})).astype(int).values)
        print(f"  n=150 训练的代理模型，在 n=200 自然分布上:")
        print(f"    代理信号 AUC   = {auc_nat:.4f}")
        print(f"    置信度 AUC     = {auc_nat_conf:.4f}")
        print(f"    辩论后分歧 AUC = {auc_nat_dis:.4f}（上界参照，需先辩论）")
        print(f"    → 代理 vs 置信度 {auc_nat-auc_nat_conf:+.4f}，"
              f"{'泛化成立' if auc_nat > auc_nat_conf else '泛化失败（可能过拟合 n=150）'}")
        generalization = {"auc_proxy": round(auc_nat, 4),
                          "auc_conf": round(auc_nat_conf, 4),
                          "auc_disagree": round(auc_nat_dis, 4),
                          "verdict": "成立" if auc_nat > auc_nat_conf else "失败"}
    except ValueError as e:
        print(f"  [无法计算] {e}")
        generalization = None
else:
    print(f"  [跳过] {NATURAL} 尚未生成（n=200 实验进行中）")
    generalization = None

# ── 结论 ──
print("\n" + "=" * 100)
print("【结论】")
print("=" * 100)
concl = [
    f"1. 辩论前代理信号最优 CV AUC = {best_cv:.4f}（{best_name}），"
    f"vs 置信度基线 0.6045（{best_cv-0.6045:+.4f}），"
    f"vs 辩论后分歧上界 0.7366（{best_cv-0.7366:+.4f}）。",
    f"2. 过拟合风险: in-sample 与 CV 差值最大达 "
    f"{max(v['overfit_gap'] for v in res_models.values()):+.4f}，"
    f"故所有结论以 5 折 CV 为准，in-sample 数值不可引用。",
    f"3. top-2 logit 差已排除: 二分类下等于 2×p_max−1，是置信度的单调变换，"
    f"AUC 必然相同，零信息增量。",
    f"4. 最强单特征: {single[0][0]}（AUC={single[0][1]:.4f}）。",
]
if generalization:
    concl.append(
        f"5. 跨分布泛化: n=150 训练的代理在 n=200 自然分布上 AUC="
        f"{generalization['auc_proxy']:.4f} vs 置信度 {generalization['auc_conf']:.4f}"
        f" → {generalization['verdict']}。")
for x in concl:
    print("  " + x)

out = {
    "n": len(df), "n_features": len(FEAT_NAMES),
    "judge_wrong_rate_pct": round(df.judge_wrong.mean() * 100, 2),
    "reference_auc": {"confidence": 0.6045, "disagreement_post_debate": 0.7366},
    "single_feature_auc": [{"feature": nm, "auc": round(a, 4)} for nm, a, _ in single],
    "models": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                   for kk, vv in v.items()} for k, v in res_models.items()},
    "best_model": best_name, "best_cv_auc": round(best_cv, 4),
    "feature_importance": [{"feature": nm, "importance": round(float(v), 4)}
                           for nm, v in imp[:15]],
    "gate_under_budget": proxy_rows,
    "proxy_vs_confidence": cmp_rows,
    "generalization_n200": generalization,
    "acc_judge_pct": round(acc_j, 2), "acc_debate_pct": round(acc_d, 2),
}
p = f"{RES}/gate_proxy_signal_result.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
pd.DataFrame(proxy_rows).to_csv(f"{RES}/gate_proxy_under_budget.csv",
                                index=False, encoding="utf-8-sig")
print(f"\n结果已写入: gate_proxy_signal_result.json / gate_proxy_under_budget.csv")
print("=" * 100)
