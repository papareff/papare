"""tweet_eval/hate 隐性攻击检测：裁决者训练 + 零样本难度探针。

为什么必须先做探针再训裁决者：
  hate 是「隐性攻击」——表层无攻击词但实际攻击，与反讽的「字面≠意图」同构。
  但难度未实测。若 1.5B 零样本就达 90%+，说明任务饱和，辩论无纠错空间，
  会重蹈 SST-2（裁决者 98%、辩论 −4.00pp）与 AG News（已否决）的覆辙。
  判据：多数类基线 57.79%，若 1.5B 零样本落在 55~75% 区间 → 困难任务，适合做泛化实验。

本脚本两个子命令：
  probe  只用 1.5B 零样本推理 n 条，测任务难度（不训裁决者，快）
  train  训练 hate 裁决者（DistilBERT/RoBERTa，同域新策略）

数据：data/raw/hate_{train,val,test}.csv（已清洗泄漏，见 clean_hate_leakage.py）
标签语义：label=1 → hateful → 框架中的 pro 票；label=0 → not hateful → con 票
"""
import os
import sys
import json
import math
import argparse
import time

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

HATE_TRAIN = "data/raw/hate_train.csv"
HATE_VAL = "data/raw/hate_val.csv"
HATE_TEST = "data/raw/hate_test.csv"
RES = "experiments/results"
LABEL_VOTE = {1: "pro", 0: "con"}


def se(pct, n):
    return math.sqrt(pct * (100 - pct) / n) if 0 < pct < 100 else 0.0


# ══════════════════════════════════════════════════
# probe：1.5B 零样本难度探针
# ══════════════════════════════════════════════════
def run_probe(n, seed, debater):
    from src.agents.debater_agent import DebaterAgent

    os.makedirs(RES, exist_ok=True)
    df = pd.read_csv(HATE_TEST)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    sub = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    n_pos = int(sub.label.sum())
    print("=" * 100)
    print(f"【难度探针】1.5B 零样本推理 hate 任务（n={len(sub)}, seed={seed}）")
    print("=" * 100)
    print(f"样本构成: hateful {n_pos} / not-hateful {len(sub)-n_pos} "
          f"(正类占比 {n_pos/len(sub)*100:.1f}%)")
    print(f"多数类基线: {max(n_pos, len(sub)-n_pos)/len(sub)*100:.2f}%")
    print(f"辩论者: {debater}", flush=True)

    cfg = {
        "name": debater,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_length": 512,
        "temperature": 0.0,        # 贪心解码，保证可复现
        "task": "hate",            # ⚠️ 见下方说明
    }
    print("\n加载辩论者…", flush=True)
    t0 = time.time()
    agent = DebaterAgent(cfg, stance="pro")
    print(f"加载完成 {time.time()-t0:.1f}s", flush=True)

    rows = []
    lats = []
    t_start = time.time()
    for i, r in sub.iterrows():
        sample = {"text": r["text"], "label": int(r["label"])}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts = time.time()
        # 用 _lens 对应的提示词做零样本判断（与辩论流程同口径）
        vote, text_out = agent.analyze(sample, examples=None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - ts
        lats.append(dt)
        gold = LABEL_VOTE[int(r["label"])]
        rows.append({
            "idx": i, "text": r["text"], "true_label": int(r["label"]),
            "gold": gold, "predicted": vote,
            "correct": int(vote == gold), "latency_sec": round(dt, 2),
        })
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t_start
            eta = el / (i + 1) * (len(sub) - i - 1)
            print(f"  [{i+1}/{len(sub)}] 滚动准确率 "
                  f"{np.mean([x['correct'] for x in rows])*100:.2f}%  "
                  f"均延迟 {np.mean(lats):.2f}s  剩余 {eta/60:.1f}min", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/hate_probe_zeroshot.csv", index=False, encoding="utf-8-sig")

    n_valid = int(d.predicted.notna().sum())
    acc = d.correct.mean() * 100
    parse_fail = len(d) - n_valid
    pro_rate = (d.predicted == "pro").mean() * 100

    print("\n" + "=" * 100)
    print("探针结果")
    print("=" * 100)
    print(f"  1.5B 零样本准确率 : {acc:.2f}%  ±{se(acc, len(d))*1.96:.2f}pp")
    print(f"  多数类基线        : {max(n_pos, len(sub)-n_pos)/len(sub)*100:.2f}%")
    print(f"  随机基线          : 50.00%")
    print(f"  投票解析失败      : {parse_fail} 条 ({parse_fail/len(d)*100:.1f}%)")
    print(f"  判 pro（hateful）率: {pro_rate:.2f}%   真实 {n_pos/len(sub)*100:.2f}%   "
          f"偏置 {pro_rate-n_pos/len(sub)*100:+.2f}pp")

    # 分类别命中
    print(f"\n  {'类别':<16} {'样本':>7} {'命中率':>9}")
    for lab, name in [(1, "真 hateful"), (0, "真 not-hateful")]:
        s = d[d.true_label == lab]
        print(f"  {name:<16} {len(s):>7} {s.correct.mean()*100:>8.2f}%")

    # 难度判定
    print(f"\n  ◆ 难度判定")
    mb = max(n_pos, len(sub) - n_pos) / len(sub) * 100
    if parse_fail / len(d) > 0.20:
        verdict = ("★ 投票解析失败率过高（>20%）——提示词的任务描述与 hate 语义不匹配，"
                   "需先改造 _lens() 支持 hate 任务，否则探针结果不可用。")
        severity = "prompt_mismatch"
    elif acc < mb - 3:
        verdict = (f"准确率 {acc:.2f}% 低于多数类基线 {mb:.2f}% → 模型未学到任务，"
                   f"需改造提示词或该任务不适合零样本。")
        severity = "below_majority"
    elif acc > 90:
        verdict = (f"准确率 {acc:.2f}% > 90% → 任务【饱和】，辩论无纠错空间，"
                   f"不适合做泛化实验（重蹈 SST-2/AG News 覆辙）。")
        severity = "saturated"
    elif mb < acc < 85:
        verdict = (f"准确率 {acc:.2f}% 落在困难区间（多数类 {mb:.2f}% ~ 85%）→ "
                   f"★ 适合做泛化实验，辩论有纠错空间。")
        severity = "suitable_hard"
    else:
        verdict = f"准确率 {acc:.2f}%，需人工判读。"
        severity = "ambiguous"
    print(f"    {verdict}")

    # 延迟外推
    lat = float(np.mean(lats))
    print(f"\n  ◆ 全量成本外推（单条 {lat:.2f}s）")
    for name, cnt in [("hate_test 2966", 2966), ("figlang twitter_test 1800", 1800),
                      ("figlang reddit_test 1800", 1800)]:
        h = lat * cnt / 3600
        print(f"    {name:<26} {h:>6.2f} h 本地(4GB)   "
              f"{h/7.0:>5.2f} h RTX 4090   ¥{h/7.0*2.80:>5.2f}")

    out = {
        "n": len(d), "seed": seed, "debater": debater,
        "zeroshot_accuracy_pct": round(acc, 2),
        "majority_baseline_pct": round(mb, 2),
        "parse_fail_n": parse_fail, "parse_fail_pct": round(parse_fail / len(d) * 100, 2),
        "pro_rate_pct": round(pro_rate, 2),
        "true_pos_rate_pct": round(n_pos / len(sub) * 100, 2),
        "bias_pp": round(pro_rate - n_pos / len(sub) * 100, 2),
        "by_class": {str(lab): round(d[d.true_label == lab].correct.mean() * 100, 2)
                     for lab in [0, 1]},
        "avg_latency_sec": round(lat, 2),
        "verdict": verdict, "severity": severity,
        "note_task_prompt": ("DebaterAgent._lens() 目前只实现了 irony/sentiment/多分类"
                             "三种视角提示词，task='hate' 会落到多分类分支（world/sports/"
                             "business/tech），投票词表不匹配 → 若 parse_fail 高即为此因"),
    }
    with open(f"{RES}/hate_probe_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: hate_probe_zeroshot.csv / hate_probe_result.json")
    print("=" * 100)


# ══════════════════════════════════════════════════
# train：hate 裁决者
# ══════════════════════════════════════════════════
def run_train(backbone, use_class_weight):
    """训练 hate 裁决者，复用 retrain_judge_twostage 的同域新策略（已验证有效）。"""
    from scripts.retrain_judge_twostage import (
        run_stage, evaluate_model, STAGE2_HP, BACKBONES)

    bb = BACKBONES[backbone]
    suffix = "cw" if use_class_weight else "nocw"
    out_dir = f"./models/judge_hate_{backbone}_{suffix}_samedomain"
    os.makedirs(RES, exist_ok=True)

    print("=" * 100)
    print(f"训练 hate 裁决者（同域新策略，骨干={bb}）")
    print("  依据：三方消融已证明同域新策略（f1_irony 选优 + 8 epoch + cosine）")
    print("        召回 49.52%→79.10%，而跨域预训练无显著净贡献，故此处只用同域策略。")
    print("=" * 100)

    hp = dict(STAGE2_HP)
    hp["max_length"] = 128   # hate 词数均值 21.4、最大 90，128 足够
    ev, _ = run_stage("hate-同域微调", os.path.basename(out_dir), bb,
                      HATE_TRAIN, HATE_VAL, hp,
                      init_from=bb, out_dir=out_dir,
                      use_class_weight=use_class_weight,
                      task_labels=("not hateful", "hateful"))

    print("\n" + "=" * 100)
    print("多测试集评测")
    print("=" * 100)
    r = evaluate_model(out_dir, os.path.basename(out_dir),
                       {"hate_test": HATE_TEST, "hate_val": HATE_VAL})
    with open(f"{RES}/retrain_hate_{backbone}_{suffix}_result.json", "w",
              encoding="utf-8") as f:
        json.dump({"val": ev, "final": r}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已写入 {RES}/retrain_hate_{backbone}_{suffix}_result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "train"])
    ap.add_argument("--n", type=int, default=100, help="probe 样本数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debater", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--backbone", default="distilbert",
                    choices=["distilbert", "roberta"])
    ap.add_argument("--no-class-weight", action="store_true")
    args = ap.parse_args()

    if args.cmd == "probe":
        run_probe(args.n, args.seed, args.debater)
    else:
        run_train(args.backbone, not args.no_class_weight)


if __name__ == "__main__":
    main()
