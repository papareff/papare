"""链式任务用：FIGLANG-twitter 裁决者训练（DistilBERT 同域新策略）。

FIGLANG 官方无 val 划分，故从 train 内部按 seed=42 切出 10% 作 val（写入独立文件）。
严禁用 test 当 val：那会让早停与 metric_for_best_model 选优看到测试集，评测虚高。
"""
import os, sys, json
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.retrain_judge_twostage import run_stage, evaluate_model

SEED = 42
SRC_TRAIN = "data/raw/figlang_twitter_train.csv"
SUB_TRAIN = "data/raw/figlang_twitter_train_sub.csv"
SUB_VAL = "data/raw/figlang_twitter_val.csv"
TEST = "data/raw/figlang_twitter_test.csv"

# ── 切分 train → sub_train / val ──
df = pd.read_csv(SRC_TRAIN)
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
shuf = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(shuf) * 0.10)
val = shuf.iloc[:n_val].reset_index(drop=True)
sub_train = shuf.iloc[n_val:].reset_index(drop=True)
sub_train[["text", "label"]].to_csv(SUB_TRAIN, index=False, encoding="utf-8")
val[["text", "label"]].to_csv(SUB_VAL, index=False, encoding="utf-8")
print(f"FIGLANG-twitter 切分: train {len(sub_train)} / val {len(val)} "
      f"(val 正类占比 {(val.label==1).mean()*100:.1f}%)")

# 泄漏复查：sub_train/val/test 三者两两无重叠
te = pd.read_csv(TEST)
s1 = set(sub_train.text.astype(str).str.strip())
s2 = set(val.text.astype(str).str.strip())
s3 = set(te.text.astype(str).str.strip())
for a, b, na, nb in [(s1, s2, "sub_train", "val"), (s1, s3, "sub_train", "test"),
                     (s2, s3, "val", "test")]:
    n = len(a & b)
    print(f"  {na} ∩ {nb} = {n}  [{'OK' if n == 0 else '泄漏!'}]")
    if n:
        raise SystemExit(f"[FAIL] {na} 与 {nb} 存在 {n} 条重叠，中止训练")

HP = {"epochs": 8, "lr": 2e-5, "bs": 16, "warmup_ratio": 0.10,
      "patience": 3, "max_length": 256}

out_dir = "./models/judge_figlang_tw_distilbert_cw_samedomain"
ev, _ = run_stage("figlang-twitter-同域微调", os.path.basename(out_dir),
                  "distilbert-base-uncased", SUB_TRAIN, SUB_VAL, HP,
                  init_from="distilbert-base-uncased", out_dir=out_dir,
                  use_class_weight=True,
                  task_labels=("not sarcastic", "sarcastic"))

r = evaluate_model(out_dir, os.path.basename(out_dir),
                   {"figlang_tw_test": TEST, "figlang_tw_val": SUB_VAL})
os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/retrain_figlang_tw_distilbert_result.json", "w",
          encoding="utf-8") as f_:
    json.dump({"split": {"train": len(sub_train), "val": len(val),
                         "test": len(te), "seed": SEED},
               "val": ev, "final": r}, f_, ensure_ascii=False, indent=2, default=str)
print("FIGLANG 裁决者训练完成")
