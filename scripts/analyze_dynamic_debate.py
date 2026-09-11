"""动态调整辩论机制的可行性分析。

用户问题：文本层用全票制修偏置、门控留给多模态融合层，那辩论机制是否全是正收益？
延伸的可行性问题：计票规则能否「动态调整」——即用一个从标定集(val)选出的规则，
在分布未知的测试集上保持接近最优？

为什么这是可行性的关键：
  此前发现计票规则依赖类别分布（多数票在均衡集 +18.79pp，全票制在自然分布 +5~6pp）。
  若规则依赖分布，则「动态调整」要成立，必须满足：val 标定的规则能泛化到 test。
  但实际部署时 test 分布未知，只能用 val 标定。若 val→test 的 regret 大，动态调整不可行。

两个互补验证：
  V1 同分布 k 折交叉验证（n=784 内部）：标定折选规则 → 测试折评估 → regret
     回答「样本量波动下规则选择是否稳定」
  V2 跨数据集标定（n=150 → n=200 → n=784）：真实异分布
     回答「分布变化下 val 标定能否泛化」——这是动态调整可行性的决定性检验

规则族（三方投票 judge/pro/con ∈ {0,1}，加权计票 score = w·votes ≥ t）：
  枚举离散权重与阈值，含纯裁决者、多数票、全票制、各类加权组合作为特例。
"""
import os
import json
import itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RES = "experiments/results"

DATASETS = {
    "n150_balanced": f"{RES}/debate_lens_n150_with.csv",
    "n200_natural": f"{RES}/natural200_debate_with.csv",
    "n784_natural": f"{RES}/roberta_B_full784_debate_with.csv",
}

print("=" * 100)
print("动态调整辩论机制的可行性分析")
print("=" * 100)

# ── 载入并统一列名 ──
data = {}
for name, p in DATASETS.items():
    if not os.path.exists(p):
        print(f"  [跳过] {name}: 文件不存在 {p}")
        continue
    df = pd.read_csv(p)
    need = {"true_label", "judge_vote", "pro_vote", "con_vote", "judge_confidence"}
    if not need.issubset(df.columns):
        print(f"  [跳过] {name}: 缺列 {need - set(df.columns)}")
        continue
    df = df.dropna(subset=["true_label", "judge_vote", "pro_vote", "con_vote"])
    df["gold"] = df["true_label"].astype(int)          # 1=pro(正类) 0=con
    df["jv"] = (df["judge_vote"] == "pro").astype(int)
    df["pv"] = (df["pro_vote"] == "pro").astype(int)
    df["cv"] = (df["con_vote"] == "pro").astype(int)
    data[name] = df.reset_index(drop=True)
    pr = df["gold"].mean()
    print(f"  {name:<16} n={len(df):>4}  正类占比 {pr*100:>5.2f}%  "
          f"裁决者判pro {df['jv'].mean()*100:>5.2f}%  "
          f"正方投pro {df['pv'].mean()*100:>5.2f}%  "
          f"反方投pro {df['cv'].mean()*100:>5.2f}%")

# ── 规则族：加权计票 score = wj*jv + wp*pv + wc*cv ≥ t ──
# 权重取 {0,1,2}，阈值取能使规则非退化的值；含纯裁决者/多数票/全票制为特例
WEIGHTS = [(1, 0, 0),      # 纯裁决者
           (0, 1, 1),      # 纯辩论方（两票）
           (1, 1, 1),      # 三方等权
           (2, 1, 1),      # 裁决者权重加倍
           (1, 2, 1),      # 正方（反讽视角）权重加倍
           (1, 1, 2),      # 反方（字面视角）权重加倍
           (1, 2, 2),      # 辩论方共同加倍
           (0, 1, 0),      # 仅正方
           (0, 0, 1)]      # 仅反方
THRESHOLDS = [1, 2, 3, 4]


def rule_predict(df, w, t):
    """按加权计票规则给出预测（1=pro 0=con）"""
    wj, wp, wc = w
    score = wj * df["jv"].values + wp * df["pv"].values + wc * df["cv"].values
    return (score >= t).astype(int)


def acc_of(pred, gold):
    return float((pred == gold).mean())


# 枚举全部 (权重,阈值) 规则，剔除退化的（全判 pro 或全判 con）
def build_rules(df):
    rules = []
    gold = df["gold"].values
    for w, t in itertools.product(WEIGHTS, THRESHOLDS):
        pred = rule_predict(df, w, t)
        pr = pred.mean()
        if pr < 0.01 or pr > 0.99:      # 退化规则（几乎全判一类）剔除
            continue
        rules.append({"w": w, "t": t, "acc": acc_of(pred, gold), "pred_pro": float(pr)})
    return rules


print("\n" + "─" * 100)
print("【各数据集最优规则】（oracle，用 test 自身选最优 —— 上界，不可部署）")
print("─" * 100)
oracle = {}
for name, df in data.items():
    rules = build_rules(df)
    if not rules:
        continue
    best = max(rules, key=lambda r: r["acc"])
    # 几个固定基准规则
    fixed = {}
    for tag, w, t in [("纯裁决者", (1, 0, 0), 1), ("多数票≥2", (1, 1, 1), 2),
                      ("全票制≥3", (1, 1, 1), 3)]:
        pred = rule_predict(df, w, t)
        fixed[tag] = acc_of(pred, df["gold"].values)
    oracle[name] = best
    print(f"  {name:<16} 最优规则 w={best['w']} t={best['t']}  "
          f"acc={best['acc']*100:.2f}%  pred_pro={best['pred_pro']*100:.1f}%")
    print(f"                 固定基准: 纯裁决者 {fixed['纯裁决者']*100:.2f}%  "
          f"多数票 {fixed['多数票≥2']*100:.2f}%  全票制 {fixed['全票制≥3']*100:.2f}%")

# ── V1 同分布 k 折交叉验证：标定稳定性与 regret ──
print("\n" + "─" * 100)
print("【V1】同分布 10 折交叉验证（每个数据集内部）——规则选择的稳定性")
print("  方法：9 折标定选最优规则 → 第 10 折评估 → regret = 该折oracle − 标定规则")
print("─" * 100)
rng = np.random.default_rng(42)
v1 = {}
for name, df in data.items():
    n = len(df)
    idx = rng.permutation(n)
    K = 10
    folds = np.array_split(idx, K)
    regrets, chosen, accs = [], [], []
    for k in range(K):
        te_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        cal = df.iloc[tr_idx]
        tst = df.iloc[te_idx]
        cal_rules = build_rules(cal)
        if not cal_rules:
            continue
        best_cal = max(cal_rules, key=lambda r: r["acc"])   # 标定集选最优
        w, t = best_cal["w"], best_cal["t"]
        pred_te = rule_predict(tst, w, t)
        acc_te = acc_of(pred_te, tst["gold"].values)
        # 该测试折的 oracle
        te_rules = build_rules(tst)
        oracle_te = max((r["acc"] for r in te_rules), default=acc_te)
        regrets.append(oracle_te - acc_te)
        chosen.append((w, t))
        accs.append(acc_te)
    v1[name] = {
        "mean_regret_pp": float(np.mean(regrets) * 100) if regrets else None,
        "max_regret_pp": float(np.max(regrets) * 100) if regrets else None,
        "mean_cv_acc": float(np.mean(accs) * 100) if accs else None,
        "rule_consistency": float(len(set(chosen)) == 1) if chosen else None,
        "n_distinct_rules": len(set(chosen)),
        "most_common_rule": max(set(chosen), key=chosen.count) if chosen else None,
    }
    r = v1[name]
    print(f"  {name:<16} 10折平均准确率 {r['mean_cv_acc']:.2f}%  "
          f"平均regret {r['mean_regret_pp']:.2f}pp  最大regret {r['max_regret_pp']:.2f}pp")
    print(f"                 选出 {r['n_distinct_rules']} 种不同规则，"
          f"众数规则 {r['most_common_rule']}  "
          f"{'✓ 规则一致' if r['rule_consistency'] else '✗ 规则不稳定'}")

# ── V2 跨数据集标定：分布变化下的泛化（决定性检验）──
print("\n" + "─" * 100)
print("【V2】跨数据集标定（真实异分布）——动态调整可行性的决定性检验")
print("  方法：在标定集选最优规则 → 直接用于目标集 → regret = 目标集oracle − 标定规则")
print("─" * 100)
v2 = []
names = list(data.keys())
for cal_name in names:
    cal_df = data[cal_name]
    cal_rules = build_rules(cal_df)
    if not cal_rules:
        continue
    best_cal = max(cal_rules, key=lambda r: r["acc"])
    w, t = best_cal["w"], best_cal["t"]
    for tgt_name in names:
        if tgt_name == cal_name:
            continue
        tgt_df = data[tgt_name]
        pred = rule_predict(tgt_df, w, t)
        acc = acc_of(pred, tgt_df["gold"].values)
        tgt_rules = build_rules(tgt_df)
        oracle_acc = max((r["acc"] for r in tgt_rules), default=acc)
        # 目标集上的固定基准（全票制，作为「静态默认规则」对照）
        acc_full3 = acc_of(rule_predict(tgt_df, (1, 1, 1), 3), tgt_df["gold"].values)
        acc_major = acc_of(rule_predict(tgt_df, (1, 1, 1), 2), tgt_df["gold"].values)
        regret = oracle_acc - acc
        v2.append({
            "calibrate_on": cal_name, "target": tgt_name,
            "cal_rule_w": w, "cal_rule_t": t,
            "cal_pos_rate": float(cal_df["gold"].mean()),
            "tgt_pos_rate": float(tgt_df["gold"].mean()),
            "tgt_acc": acc, "tgt_oracle": oracle_acc, "regret_pp": regret * 100,
            "tgt_full3_acc": acc_full3, "tgt_majority_acc": acc_major,
            "beats_full3": acc > acc_full3, "beats_majority": acc > acc_major,
        })
        print(f"  在 {cal_name:<15}(正类{cal_df['gold'].mean()*100:.0f}%) 标定 w={w} t={t} "
              f"→ 用于 {tgt_name:<15}(正类{tgt_df['gold'].mean()*100:.0f}%): "
              f"acc {acc*100:.2f}%  oracle {oracle_acc*100:.2f}%  regret {regret*100:+.2f}pp")

# ── 汇总判定 ──
print("\n" + "=" * 100)
print("【可行性判定】")
print("=" * 100)
v2df = pd.DataFrame(v2)
mean_regret = v2df["regret_pp"].mean()
max_regret = v2df["regret_pp"].max()
# 标定规则 vs 静态全票制：动态调整是否真的优于「固定用全票制」
dyn_better = int(v2df["tgt_acc"].gt(v2df["tgt_full3_acc"]).sum())
n_pairs = len(v2df)
print(f"  V1 同分布：规则选择{'稳定' if all(v['rule_consistency'] for v in v1.values()) else '不稳定'}"
      f"（各数据集 10 折选出的规则种类：{[v['n_distinct_rules'] for v in v1.values()]}）")
print(f"  V2 跨分布：{n_pairs} 组标定→目标，平均 regret {mean_regret:.2f}pp，最大 {max_regret:.2f}pp")
print(f"           动态标定规则优于静态全票制的比例：{dyn_better}/{n_pairs}")
verdict = []
if max_regret > 5:
    verdict.append(f"✗ 跨分布最大 regret {max_regret:.2f}pp 偏大 → val 标定的规则在异分布 test 上会显著失效")
else:
    verdict.append(f"✓ 跨分布最大 regret {max_regret:.2f}pp 可接受 → val 标定可泛化")
if dyn_better <= n_pairs / 2:
    verdict.append("✗ 动态标定不优于静态全票制 → 「动态调整计票规则」无实际收益，应固定用全票制")
else:
    verdict.append("✓ 动态标定多数情况优于静态全票制 → 动态调整有价值")
for v in verdict:
    print(f"  {v}")

out = {
    "datasets": {k: {"n": len(v), "pos_rate": float(v["gold"].mean()),
                     "judge_pro_rate": float(v["jv"].mean()),
                     "debater_pro_rate": float(v["pv"].mean())}
                 for k, v in data.items()},
    "oracle_rules": {k: {"w": list(v["w"]), "t": v["t"], "acc": v["acc"]}
                     for k, v in oracle.items()},
    "v1_kfold": v1,
    "v2_cross_dataset": v2,
    "summary": {"v2_mean_regret_pp": float(mean_regret),
                "v2_max_regret_pp": float(max_regret),
                "dyn_beats_full3": f"{dyn_better}/{n_pairs}",
                "verdict": verdict},
}
os.makedirs(RES, exist_ok=True)
with open(f"{RES}/dynamic_debate_feasibility.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n结果已写入 {RES}/dynamic_debate_feasibility.json")
