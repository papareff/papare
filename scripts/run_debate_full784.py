"""全量测试集辩论验证（n=784，参数化裁决者）。

用途：在完整 SemEval 测试集上验证「重训后裁决者 + 视角分化辩论」的真实增益，
     并扫描门控阈值。替代 run_natural_dist_n200.py 的固定抽样逻辑。

为什么要跑全量：
  n=200 对 +13.43pp 增益的功效 85.2%（已推翻事后分层估算），
  但对门控 +4.50pp 的功效仅 17.0%；n=784 提升到 51.3%，仍需谨慎表述。

用法：
  python scripts/run_debate_full784.py --judge models/judge_roberta_cw_samedomain_ablation --out roberta_B
  python scripts/run_debate_full784.py --judge models/irony_distilbert --out baseline_A --n 200  # 小规模试跑

贪心解码（temperature=0.0），可复现。1.5B 辩论实测（本机 RTX 3050 Laptop 4GB）：
  DistilBERT 裁决者 24.45s/条 → n=784 约 5.3h
  RoBERTa 裁决者   26.08s/条 → n=784 约 5.7h
  RTX 4090 约 0.76–0.81h
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import time
import numpy as np
import pandas as pd
import torch
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

DEFAULT_TEST_CSV = "data/raw/irony_test.csv"
OUT_DIR = "experiments/results"
LABEL_VOTE = {1: "pro", 0: "con"}   # pro = 正类(1), con = 负类(0)

# 任务感知的类别展示名（用于日志与列名口径说明）。
# 三个二分类任务同构：label=1 均为「需识别的隐含意义」→ pro 票。
TASK_LABEL_NAMES = {
    "irony": {"pos": "反讽", "neg": "非反讽", "max_length": 128},
    "hate": {"pos": "攻击性", "neg": "非攻击", "max_length": 128},
    "figlang": {"pos": "讽刺", "neg": "非讽刺", "max_length": 256},
}


def se(pct, n):
    return math.sqrt(pct * (100 - pct) / n) if 0 < pct < 100 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="./models/irony_distilbert",
                    help="裁决者模型目录")
    ap.add_argument("--out", default="full784",
                    help="输出文件名前缀")
    ap.add_argument("--n", type=int, default=784,
                    help="样本数，默认全量 784")
    ap.add_argument("--seed", type=int, default=42,
                    help="抽样种子（n<全量时生效；全量时忽略）")
    ap.add_argument("--mode", default="natural", choices=["natural", "balanced"],
                    help="natural=按原始分布随机抽样；balanced=按类别均衡分层抽样")
    ap.add_argument("--debater", default="Qwen/Qwen2-1.5B-Instruct",
                    help="辩论者模型")
    ap.add_argument("--rounds", type=int, default=1, help="辩论轮数")
    ap.add_argument("--max-memory-gpu", default=None,
                    help="7B 跨设备卸载时的显存上限，如 3GiB；1.5B 留空")
    ap.add_argument("--task", default="irony", choices=list(TASK_LABEL_NAMES.keys()),
                    help="任务名，决定辩论者视角提示词与类名口径")
    ap.add_argument("--test-csv", default=DEFAULT_TEST_CSV,
                    help="测试集路径（默认 irony_test.csv）")
    ap.add_argument("--max-length", type=int, default=None,
                    help="裁决者截断长度；缺省按任务取默认（irony/hate=128, figlang=256）")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    task_meta = TASK_LABEL_NAMES[args.task]
    POS, NEG = task_meta["pos"], task_meta["neg"]
    max_length = args.max_length or task_meta["max_length"]

    if not os.path.isdir(args.judge):
        raise SystemExit(f"[FAIL] 裁决者目录不存在: {args.judge}")

    # ── 抽样 ──
    test_csv = args.test_csv
    if not os.path.exists(test_csv):
        raise SystemExit(f"[FAIL] 测试集不存在: {test_csv}")
    test_df = pd.read_csv(test_csv)
    test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce")
    test_df = test_df.dropna(subset=["label", "text"]).copy()
    test_df["label"] = test_df["label"].astype(int)
    test_df = test_df.reset_index(drop=True)

    n_full = len(test_df)
    n_run = min(args.n, n_full)
    if n_run >= n_full:
        samples = test_df.to_dict("records")
        print(f"全量运行: {n_full} 条（忽略 --seed/--mode）", flush=True)
    elif args.mode == "balanced":
        per = n_run // 2
        samples = (test_df.groupby("label", group_keys=False)
                   .apply(lambda g: g.sample(n=min(len(g), per),
                                             random_state=args.seed))
                   .sample(frac=1.0, random_state=args.seed)
                   .reset_index(drop=True).to_dict("records"))
        print(f"均衡分层抽样: {len(samples)} 条（seed={args.seed}）", flush=True)
    else:
        samples = (test_df.sample(n=n_run, random_state=args.seed)
                   .reset_index(drop=True).to_dict("records"))
        print(f"自然分布随机抽样: {len(samples)} 条（seed={args.seed}）", flush=True)

    n_pro = sum(1 for s in samples if int(s["label"]) == 1)
    N = len(samples)
    print(f"样本构成: {POS} {n_pro} / {NEG} {N - n_pro}"
          f"（{POS}占比 {n_pro / N * 100:.1f}%）", flush=True)
    print(f"完整测试集自然占比: {(test_df.label == 1).mean() * 100:.1f}%", flush=True)
    print(f"任务: {args.task}   裁决者截断长度: {max_length}   测试集: {test_csv}",
          flush=True)

    # ── 配置 ──
    with open("configs/base.yaml", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    config = dict(base_config)
    config["task"] = args.task
    config["judge"] = {"name": args.judge, "device": "cuda"
                       if torch.cuda.is_available() else "cpu",
                       "max_length": max_length}
    debate_cfg = {
        "rounds": args.rounds,
        "neutral": True,          # 立场中立化 + 视角分化（_lens）
        "debater_model": args.debater,
    }
    if args.max_memory_gpu:
        debate_cfg["max_memory"] = {0: args.max_memory_gpu, "cpu": "25GiB"}
    config["debate"] = debate_cfg

    print(f"\n加载管线（辩论者={args.debater}, 裁决者={args.judge}）…", flush=True)
    t_load = time.time()
    pipeline = InferencePipeline(config, use_debate=True, retriever=None, num_shots=0)
    print(f"加载完成，耗时 {time.time() - t_load:.1f}s", flush=True)

    def correct_of(tl, pred):
        return pred == LABEL_VOTE[int(tl)]

    # ══════════ 1) 视角分化辩论 ══════════
    print(f"\n===== 视角分化辩论（n={N}）=====", flush=True)
    rows, lats = [], []
    t_start = time.time()
    for i, s in enumerate(samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        r = pipeline.run(s)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        lats.append(dt)
        v = r.get("debate_votes", {}) or {}
        rows.append({
            "sample_id": i, "text": s["text"], "true_label": int(s["label"]),
            "gold": LABEL_VOTE[int(s["label"])],
            "predicted": r["decision"], "judge_vote": r.get("judge_vote"),
            "judge_confidence": r.get("judge_confidence"),
            "pro_vote": v.get("pro"), "con_vote": v.get("con"),
            "votes_agree": int(v.get("pro") == v.get("con")),
            "latency_sec": round(dt, 2),
            "correct": int(correct_of(s["label"], r["decision"])),
        })
        # 增量落盘：长任务中断也不丢已完成部分
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(
                f"{OUT_DIR}/{args.out}_debate_with.partial.csv",
                index=False, encoding="utf-8-sig")
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t_start
            eta = el / (i + 1) * (N - i - 1)
            run_acc = np.mean([x["correct"] for x in rows]) * 100
            print(f"  [{i + 1}/{N}] 滚动准确率 {run_acc:.2f}%  "
                  f"均延迟 {np.mean(lats):.2f}s  已用 {el / 60:.1f}min  "
                  f"剩余 {eta / 60:.1f}min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/{args.out}_debate_with.csv", index=False, encoding="utf-8-sig")
    partial = f"{OUT_DIR}/{args.out}_debate_with.partial.csv"
    if os.path.exists(partial):
        os.remove(partial)

    # ══════════ 2) 纯裁决者基线 ══════════
    print(f"\n===== 纯裁决者基线（n={N}）=====", flush=True)
    base_rows = []
    for i, s in enumerate(samples):
        res = pipeline.judge.make_decision(s, debate_votes=None)
        base_rows.append({
            "sample_id": i, "true_label": int(s["label"]),
            "gold": LABEL_VOTE[int(s["label"])],
            "predicted": res["decision"], "judge_confidence": res["judge_confidence"],
            "correct": int(correct_of(s["label"], res["decision"])),
        })
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(f"{OUT_DIR}/{args.out}_judge_only.csv",
                   index=False, encoding="utf-8-sig")

    # ══════════ 3) 统计汇总 ══════════
    acc_d = df.correct.mean() * 100
    acc_j = base_df.correct.mean() * 100
    delta = acc_d - acc_j

    merged = df.merge(
        base_df[["sample_id", "correct"]].rename(columns={"correct": "jc"}),
        on="sample_id")
    disc = merged[merged.correct != merged.jc]
    b_cnt = int((disc.correct == 1).sum())   # 辩论对/裁决者错
    c_cnt = int((disc.jc == 1).sum())        # 裁决者对/辩论错
    chi2 = ((abs(b_cnt - c_cnt) - 1) ** 2 / (b_cnt + c_cnt)) if (b_cnt + c_cnt) else 0.0
    se_pair = (math.sqrt((b_cnt + c_cnt) - (b_cnt - c_cnt) ** 2 / (b_cnt + c_cnt))
               / N * 100) if (b_cnt + c_cnt) else 0.0

    print("\n" + "=" * 96)
    print(f"全量辩论验证结果  task={args.task}  out={args.out}  judge={args.judge}  "
          f"n={N}  {POS}占比 {n_pro / N * 100:.1f}%")
    print("=" * 96)
    print("① 准确率对照")
    print(f"   视角分化辩论 : {acc_d:.2f}%  ±{se(acc_d, N) * 1.96:.2f}pp")
    print(f"   纯裁决者     : {acc_j:.2f}%  ±{se(acc_j, N) * 1.96:.2f}pp")
    print(f"   辩论增益     : {delta:+.2f}pp  ±{se_pair * 1.96:.2f}pp")
    print(f"   McNemar b={b_cnt} c={c_cnt} χ²={chi2:.3f} → "
          f"{'显著' if chi2 > 3.841 else '不显著'}（临界 3.841, α=0.05）")

    print("\n② 分类别命中")
    print(f"   {'类别':<12} {'裁决者':>9} {'辩论':>9} {'差值':>10} {'样本数':>8}")
    cls = []
    for lab, name in [(1, f"真{POS}"), (0, f"真{NEG}")]:
        s1 = merged[merged.true_label == lab]
        aj = s1.jc.mean() * 100
        ad = s1.correct.mean() * 100
        cls.append({"class": name, "n": len(s1), "acc_judge": round(aj, 2),
                    "acc_debate": round(ad, 2), "delta_pp": round(ad - aj, 2)})
        print(f"   {name:<12} {aj:>8.2f}% {ad:>8.2f}% {ad - aj:>+9.2f}pp {len(s1):>8}")
    # 临界正类占比 π*：低于此值辩论净增益转负（与 diagnose_gain_collapse.py 同源）
    g = cls[0]["delta_pp"]
    h = cls[1]["delta_pp"]
    pi_star = (-h / (g - h) * 100) if (g - h) != 0 else None

    print("\n③ 判决分布与偏置")
    print(f"   真实{POS}占比 {n_pro / N * 100:.1f}%   "
          f"辩论判 pro {(df.predicted == 'pro').mean() * 100:.1f}%   "
          f"裁决者判 pro {(base_df.predicted == 'pro').mean() * 100:.1f}%")
    print(f"   辩论方 pro 偏置 {(df.predicted == 'pro').mean() * 100 - n_pro / N * 100:+.2f}pp   "
          f"裁决者 pro 偏置 {(base_df.predicted == 'pro').mean() * 100 - n_pro / N * 100:+.2f}pp")
    print(f"   双方投票一致率 {df.votes_agree.mean() * 100:.1f}%")
    bias_d = (df.predicted == "pro").mean() * 100 - n_pro / N * 100
    bias_j = (base_df.predicted == "pro").mean() * 100 - n_pro / N * 100
    print(f"   偏置互补性: {'反向（互补，辩论应有增益）' if bias_d * bias_j < 0 else '同向（叠加，辩论增益可能消失）'}")

    print("\n④ 门控阈值扫描")
    lat_d = float(np.mean(lats))
    print(f"   {'τ':>6} {'准确率':>9} {'触发率':>8} {'加权延迟':>10} {'vs纯裁决者':>11} {'延迟节省':>9}")
    gate_rows, best = [], None
    for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]:
        low = df.judge_confidence < tau
        pred = np.where(low, df.predicted, df.judge_vote)
        a = (pd.Series(pred, index=df.index) == df.gold).mean() * 100
        trig = low.mean() * 100
        wl = trig / 100 * lat_d + (1 - trig / 100) * 0.01
        gate_rows.append({"tau": tau, "accuracy": round(a, 2),
                          "trigger_rate": round(trig, 2),
                          "weighted_latency": round(wl, 2),
                          "vs_judge_pp": round(a - acc_j, 2),
                          "vs_debate_pp": round(a - acc_d, 2)})
        print(f"   {tau:>6.2f} {a:>8.2f}% {trig:>7.1f}% {wl:>9.2f}s "
              f"{a - acc_j:>+10.2f}pp {(1 - wl / lat_d) * 100:>8.0f}%")
        # 效率约束选优：触发率 ≤60% 中取准确率最高
        if trig <= 60.0 and (best is None or a > best["accuracy"]):
            best = {"tau": tau, "accuracy": a, "trigger_rate": trig,
                    "weighted_latency": wl}
    pd.DataFrame(gate_rows).to_csv(f"{OUT_DIR}/{args.out}_gate_scan.csv",
                                   index=False, encoding="utf-8-sig")
    if best:
        print(f"   ★ 效率约束内最优（触发率≤60%） τ={best['tau']:.2f}: "
              f"{best['accuracy']:.2f}%  触发率 {best['trigger_rate']:.1f}%  "
              f"延迟节省 {(1 - best['weighted_latency'] / lat_d) * 100:.0f}%")
        print(f"     vs 纯裁决者 {best['accuracy'] - acc_j:+.2f}pp   "
              f"vs 全程辩论 {best['accuracy'] - acc_d:+.2f}pp"
              f"{'  ← 门控超过全程辩论' if best['accuracy'] > acc_d else ''}")
    else:
        print("   ★ 触发率≤60% 内无可行操作点")

    pd.DataFrame(cls).to_csv(f"{OUT_DIR}/{args.out}_by_class.csv",
                             index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "out": args.out, "judge": args.judge, "debater": args.debater,
        "n": N, "mode": args.mode if N < n_full else "full",
        "task": args.task, "test_csv": test_csv, "max_length": max_length,
        "irony_ratio_pct": round(n_pro / N * 100, 2), "rounds": args.rounds,
        "pos_rate_pct": round(n_pro / N * 100, 2),
        "pos_class_name": POS, "neg_class_name": NEG,
        "acc_debate_pct": round(acc_d, 2), "acc_judge_pct": round(acc_j, 2),
        "delta_pp": round(delta, 2), "delta_ci95_pp": round(se_pair * 1.96, 2),
        "mcnemar_b": b_cnt, "mcnemar_c": c_cnt, "mcnemar_chi2": round(chi2, 3),
        "significant": bool(chi2 > 3.841),
        "gain_pos_pp": cls[0]["delta_pp"], "harm_neg_pp": cls[1]["delta_pp"],
        # 兼容旧分析脚本的字段名（irony 专用口径）
        "gain_irony_pp": cls[0]["delta_pp"], "harm_nonirony_pp": cls[1]["delta_pp"],
        "critical_pi_pct": round(pi_star, 2) if pi_star is not None else None,
        "debate_bias_pp": round(bias_d, 2), "judge_bias_pp": round(bias_j, 2),
        "bias_complementary": bool(bias_d * bias_j < 0),
        "vote_agree_pct": round(df.votes_agree.mean() * 100, 2),
        "avg_latency_sec": round(lat_d, 2),
        "best_tau": best["tau"] if best else None,
        "best_gate_acc_pct": round(best["accuracy"], 2) if best else None,
        "best_gate_trigger_pct": round(best["trigger_rate"], 2) if best else None,
    }]).to_csv(f"{OUT_DIR}/{args.out}_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 96)
    print(f"结果文件: {args.out}_debate_with.csv / {args.out}_judge_only.csv /")
    print(f"          {args.out}_gate_scan.csv / {args.out}_by_class.csv / {args.out}_summary.csv")
    print("=" * 96)


if __name__ == "__main__":
    main()
