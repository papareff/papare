"""链式启动器：等 n=784 全量实验落盘后，自动依次执行 hate 与 FIGLANG 的本地任务。

为什么需要链式：本机仅 4GB 显存，n=784 全量辩论（RoBERTa 裁决者 + 1.5B 辩论者，
约 3.7GB）运行期间无法并发加载第二个 1.5B 实例（OOM），必须串行。

队列（全部为本地可行任务，重推理留给服务器）：
  1. 等待 experiments/results/roberta_B_full784_summary.csv 出现（n=784 完成标志）
  2. hate 难度探针：1.5B 零样本 n=100（约 37min）→ 判定 hate 是否困难任务
  3. hate 裁决者训练：DistilBERT 同域新策略（约 10min）
  4. FIGLANG-twitter 裁决者训练：DistilBERT 同域新策略（约 6min）

每步的输出都带 [CHAIN] 前缀与时间戳，便于事后审计。
任何一步失败不中断后续步骤（除非 GPU 不可用）。
"""
import os
import sys
import time
import subprocess
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.path.join(ROOT, "envs", "debate_agent", "Scripts", "python.exe")
RES = os.path.join(ROOT, "experiments", "results")
WAIT_FILE = os.path.join(RES, "roberta_B_full784_summary.csv")
POLL_SEC = 60
WAIT_TIMEOUT_H = 3.5

ENV = dict(os.environ)
ENV["PYTHONPATH"] = ""
ENV["HF_HOME"] = r"E:\huggingface_cache"
ENV["TRANSFORMERS_OFFLINE"] = "1"
ENV["HF_HUB_OFFLINE"] = "1"
ENV["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[CHAIN {ts}] {msg}", flush=True)


def wait_for_full784():
    log(f"等待 n=784 全量实验完成（标志文件: {os.path.basename(WAIT_FILE)}）…")
    t0 = time.time()
    while not os.path.exists(WAIT_FILE):
        if (time.time() - t0) / 3600 > WAIT_TIMEOUT_H:
            log(f"[WARN] 等待超过 {WAIT_TIMEOUT_H}h 仍未见完成标志，继续执行队列（GPU 可能已空闲）")
            return False
        time.sleep(POLL_SEC)
    log(f"检测到完成标志，等待 90s 确保进程完全退出、显存释放…")
    time.sleep(90)
    return True


def run_step(name, args, timeout_h=2.0):
    log(f"▶ 开始: {name}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PY, "-u"] + args, cwd=ROOT, env=ENV,
            timeout=int(timeout_h * 3600),
            capture_output=False,
        )
        dt = (time.time() - t0) / 60
        if proc.returncode == 0:
            log(f"✓ 完成: {name}  ({dt:.1f} min)")
            return True
        log(f"✗ 失败: {name}  returncode={proc.returncode}  ({dt:.1f} min)")
        return False
    except subprocess.TimeoutExpired:
        log(f"✗ 超时: {name} (> {timeout_h}h)")
        return False
    except Exception as e:
        log(f"✗ 异常: {name}  {type(e).__name__}: {e}")
        return False


def main():
    log("=" * 88)
    log("链式任务队列启动")
    log("=" * 88)

    wait_for_full784()

    results = {}

    # ── 步骤 2: hate 难度探针 ──
    results["hate_probe"] = run_step(
        "hate 难度探针（1.5B 零样本 n=100）",
        ["scripts/run_hate_task.py", "probe", "--n", "100", "--seed", "42"],
        timeout_h=1.5)

    # ── 步骤 3: hate 裁决者训练（DistilBERT 同域新策略）──
    results["hate_judge_distilbert"] = run_step(
        "hate 裁决者训练（DistilBERT 同域新策略）",
        ["scripts/run_hate_task.py", "train", "--backbone", "distilbert"],
        timeout_h=1.0)

    # ── 步骤 4: FIGLANG-twitter 裁决者训练 ──
    # FIGLANG 官方只有 train/test 两个划分，无 val。而 run_stage 需要 val 做早停与选优。
    # 解决方案：从 train 内部按固定 seed 切出 10% 作 val，写入独立文件。
    #   ⚠️ 绝不能拿 test 当 val —— 那会让早停/选优看到测试集，评测结果虚高。
    # 拼接文本词数均值 111、最大 425，max_length 需 256（128 会截断上下文）。
    figlang_driver = os.path.join(ROOT, "scripts", "_chain_figlang_train.py")
    with open(figlang_driver, "w", encoding="utf-8") as f:
        f.write('''"""链式任务用：FIGLANG-twitter 裁决者训练（DistilBERT 同域新策略）。

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
''')
    results["figlang_judge"] = run_step(
        "FIGLANG-twitter 裁决者训练（DistilBERT 同域新策略）",
        ["scripts/_chain_figlang_train.py"], timeout_h=1.5)

    log("=" * 88)
    log("链式队列结束:")
    for k, v in results.items():
        log(f"  {k:<28} {'成功' if v else '失败'}")
    log("=" * 88)


if __name__ == "__main__":
    main()
