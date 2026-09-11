"""门控机制离线分析（零 GPU 成本）。

依据：反讽辩论实验结果已逐条记录裁决者置信度 judge_confidence、裁决者独立判决 judge_vote、
辩论后最终判决 predicted、真实标签 true_label。因此可直接离线扫描门控阈值 τ：

  门控策略：conf >= τ → 直接采纳裁决者判决（不启动辩论，成本≈0）
            conf <  τ → 启动辩论并采纳辩论后判决（成本=辩论延迟）

输出：
  1) 各 τ 下的准确率、辩论触发率、加权平均延迟
  2) 低置信子集上「辩论 vs 裁决者」的局部对比（门控是否有正增益的关键判据）
  3) 高置信子集上「辩论 vs 裁决者」的对比（验证辩论是否在高置信处帮倒忙）
"""
import os
import pandas as pd
import numpy as np

RES = "experiments/results"
LABEL_VOTE = {1: "pro", 0: "con"}


def load(fname):
    p = os.path.join(RES, fname)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    need = {"true_label", "predicted", "judge_confidence"}
    if not need.issubset(df.columns):
        print(f"  跳过 {fname}：缺少字段 {need - set(df.columns)}")
        return None
    return df


def acc(pred_series, true_series):
    gold = true_series.map(LABEL_VOTE)
    return (pred_series == gold).mean()


def analyse(name, df, judge_col="judge_vote"):
    print("\n" + "=" * 78)
    print(f"【{name}】 n={len(df)}")
    print("=" * 78)

    df = df.copy()
    gold = df["true_label"].map(LABEL_VOTE)
    has_judge = judge_col in df.columns

    # 基线
    acc_debate = (df["predicted"] == gold).mean() * 100
    if has_judge:
        acc_judge = (df[judge_col] == gold).mean() * 100
    else:
        acc_judge = float("nan")
    lat_debate = df["latency_sec"].mean() if "latency_sec" in df.columns else float("nan")

    print(f"  全程辩论准确率      : {acc_debate:.2f}%   平均延迟 {lat_debate:.2f}s")
    if has_judge:
        print(f"  纯裁决者(无辩论)准确率: {acc_judge:.2f}%   平均延迟 ~0.01s")

    conf = df["judge_confidence"]
    print(f"  置信度分布          : min={conf.min():.3f} p25={conf.quantile(.25):.3f} "
          f"中位={conf.median():.3f} p75={conf.quantile(.75):.3f} max={conf.max():.3f}")
    print(f"  置信度<0.6 的样本占比: {(conf < 0.6).mean()*100:.1f}%")
    print(f"  置信度<0.8 的样本占比: {(conf < 0.8).mean()*100:.1f}%")

    if not has_judge:
        print("  （无 judge_vote 列，跳过门控扫描）")
        return

    # ---- 门控阈值扫描 ----
    print("\n  门控阈值扫描（conf>=τ 用裁决者，conf<τ 用辩论）：")
    print("  " + "-" * 74)
    print(f"  {'τ':>6} {'准确率':>9} {'辩论触发率':>11} {'加权延迟':>10} "
          f"{'vs纯裁决者':>11} {'vs全程辩论':>11}")
    print("  " + "-" * 74)
    rows = []
    for tau in [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01]:
        low = conf < tau                      # 启动辩论
        pred = np.where(low, df["predicted"], df[judge_col])
        a = (pd.Series(pred, index=df.index) == gold).mean() * 100
        trig = low.mean() * 100
        lat = trig / 100 * lat_debate + (1 - trig / 100) * 0.01
        rows.append({"tau": tau, "accuracy": a, "trigger_rate": trig, "weighted_latency": lat})
        print(f"  {tau:>6.2f} {a:>8.2f}% {trig:>10.1f}% {lat:>9.2f}s "
              f"{a-acc_judge:>+10.2f}pp {a-acc_debate:>+10.2f}pp")
    print("  " + "-" * 74)

    best = max(rows, key=lambda r: r["accuracy"])
    print(f"\n  ★ 最优阈值 τ={best['tau']:.2f}：准确率 {best['accuracy']:.2f}%"
          f"（触发率 {best['trigger_rate']:.1f}%，加权延迟 {best['weighted_latency']:.2f}s）")
    print(f"    相对纯裁决者 {best['accuracy']-acc_judge:+.2f}pp，"
          f"相对全程辩论 {best['accuracy']-acc_debate:+.2f}pp，"
          f"延迟降至全程辩论的 {best['weighted_latency']/lat_debate*100:.1f}%")

    # ---- 分子集诊断：辩论在哪里有用、哪里有害 ----
    print("\n  分子集诊断（辩论 vs 裁决者）：")
    print("  " + "-" * 74)
    print(f"  {'子集':<22} {'n':>5} {'裁决者':>9} {'辩论后':>9} {'辩论增益':>10}")
    print("  " + "-" * 74)
    subsets = [
        ("全部样本", np.ones(len(df), bool)),
        ("高置信 conf>=0.9", (conf >= 0.9).values),
        ("中置信 0.7~0.9", ((conf >= 0.7) & (conf < 0.9)).values),
        ("低置信 0.6~0.7", ((conf >= 0.6) & (conf < 0.7)).values),
        ("极低置信 conf<0.6", (conf < 0.6).values),
    ]
    for label, mask in subsets:
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<22} {n:>5} {'—':>9} {'—':>9} {'—':>10}")
            continue
        sub = df[mask]; g = gold[mask]
        aj = (sub[judge_col] == g).mean() * 100
        ad = (sub["predicted"] == g).mean() * 100
        print(f"  {label:<22} {n:>5} {aj:>8.2f}% {ad:>8.2f}% {ad-aj:>+9.2f}pp")
    print("  " + "-" * 74)

    # 辩论翻转裁决者的案例统计
    flipped = df[df["predicted"] != df[judge_col]]
    if len(flipped):
        g = flipped["true_label"].map(LABEL_VOTE)
        good = (flipped["predicted"] == g).sum()
        bad = (flipped[judge_col] == g).sum()
        print(f"\n  辩论翻转裁决者的样本: {len(flipped)} 条"
              f"（占 {len(flipped)/len(df)*100:.1f}%）")
        print(f"    其中辩论改对 {good} 条，辩论改错 {bad} 条，"
              f"净效果 {good-bad:+d} 条")
        fc = flipped["judge_confidence"]
        print(f"    被翻转样本的置信度: 均值 {fc.mean():.3f}，中位 {fc.median():.3f}，"
              f"最高 {fc.max():.3f}")
        print(f"    翻转发生在 conf>=0.9 的条数: {int((fc>=0.9).sum())}"
              f"（这些是高置信却被辩论带偏的案例）")

    return rows


print("门控机制离线分析")
print("数据来源：已完成的辩论消融实验逐条结果（含裁决者置信度）")

datasets = [
    ("反讽·立场中立化辩论", "irony_debate_neutral_with.csv"),
    ("反讽·预设立场辩论", "irony_debate_with.csv"),
    # 注：SST-2 试点结果 pilot50_with_debate.csv 的 judge_vote 列为空（记录缺陷），
    # 无法用于门控对比，故排除；其辩论消融结论（94% vs 98%）另见实验记录文档。
]

summary = {}
for name, f in datasets:
    df = load(f)
    if df is None:
        continue
    rows = analyse(name, df)
    if rows:
        summary[name] = rows

if summary:
    print("\n\n" + "=" * 78)
    print("汇总：门控相对两种极端策略的收益")
    print("=" * 78)
    for name, rows in summary.items():
        best = max(rows, key=lambda r: r["accuracy"])
        # τ=0 → 从不启动辩论 = 纯裁决者；τ=1.01 → 始终辩论 = 全程辩论
        pure_judge = next(r for r in rows if r["tau"] == 0.0)
        full_debate = next(r for r in rows if r["tau"] == 1.01)
        print(f"\n  {name}")
        print(f"    纯裁决者(τ=0)   : {pure_judge['accuracy']:.2f}%  延迟 ~0.01s")
        print(f"    全程辩论(τ=1.01): {full_debate['accuracy']:.2f}%  延迟 {full_debate['weighted_latency']:.2f}s")
        print(f"    门控最优        : {best['accuracy']:.2f}%  延迟 {best['weighted_latency']:.2f}s  (τ={best['tau']:.2f})")
        print(f"    门控 vs 纯裁决者: {best['accuracy']-pure_judge['accuracy']:+.2f}pp   "
              f"门控 vs 全程辩论: {best['accuracy']-full_debate['accuracy']:+.2f}pp")
        print(f"    成本节省: 加权延迟仅为全程辩论的 "
              f"{best['weighted_latency']/max(full_debate['weighted_latency'],1e-9)*100:.1f}%")

    # 落盘汇总表
    all_rows = []
    for name, rows in summary.items():
        for r in rows:
            all_rows.append({"dataset": name, **r})
    out = pd.DataFrame(all_rows)
    os.makedirs(RES, exist_ok=True)
    out.to_csv(os.path.join(RES, "gate_threshold_scan.csv"), index=False, encoding="utf-8-sig")
    print(f"\n阈值扫描明细已保存到 {RES}/gate_threshold_scan.csv")
