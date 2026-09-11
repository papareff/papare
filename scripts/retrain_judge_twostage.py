"""两阶段裁决者重训练：Misra 域适应预训练 → SemEval 任务微调。

方案 A（用户选定）+ 双骨干对照：
  骨干 1: distilbert-base-uncased  (66M)  —— 与现有结果可比
  骨干 2: roberta-base             (125M) —— 容量对照

阶段1：Misra 21281 新闻标题讽刺（域适应，学"讽刺=字面与事实背离"的通用表征）
阶段2：SemEval 2575 推特反讽（任务微调，适配推特语域与 #irony 标注机制）
评测 ：SemEval test 784（主）+ Misra test 2661（跨域诊断）

相对原 finetune_irony.py 的四项改进：
  1. 类别权重（解决反讽召回仅 49.33%）
  2. warmup 比例化（原 50 步 / 81 步每epoch = 62%，过高）
  3. 余弦学习率衰减 + 更多 epoch
  4. 早停耐心度提升，避免 epoch 1 即停导致数据未充分利用

RoBERTa-base 不在本地缓存，服务器端首次运行会自动下载（约 500MB）。
"""
import os
import re
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)

MISRA_TRAIN = "data/raw/misra_train.csv"
MISRA_VAL = "data/raw/misra_val.csv"
MISRA_TEST = "data/raw/misra_test.csv"
SEMEVAL_TRAIN = "data/raw/irony_train.csv"
SEMEVAL_VAL = "data/raw/irony_val.csv"
SEMEVAL_TEST = "data/raw/irony_test.csv"

BACKBONES = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}

# 各任务的标签语义 (负类名, 正类名)。默认 irony，保持对现有调用的完全兼容。
TASK_LABELS = {
    "irony": ("not ironic", "ironic"),
    "hate": ("not hateful", "hateful"),
    "sarcasm": ("not sarcastic", "sarcastic"),
}

# 阶段1/阶段2 的超参（分别调，因样本量差 8 倍）
STAGE1_HP = {
    "epochs": 4, "lr": 5e-5, "bs": 32, "warmup_ratio": 0.06,
    "patience": 2, "max_length": 64,   # 新闻标题均 9.8 词，64 足够
}
STAGE2_HP = {
    "epochs": 8, "lr": 2e-5, "bs": 16, "warmup_ratio": 0.10,
    "patience": 3, "max_length": 128,  # 推特均 13.7 词，含话题标签
}


def compute_class_weights(labels):
    """逆频率类别权重，用于纠正反讽类召回不足。"""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2).astype(float)
    w = len(labels) / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32), counts


class WeightedTrainer(Trainer):
    """带类别权重的 Trainer（原脚本无此项，是召回 49.33% 的成因之一）。"""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            loss_f = torch.nn.CrossEntropyLoss(weight=cw)
        else:
            loss_f = torch.nn.CrossEntropyLoss()
        loss = loss_f(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_irony": f1_score(labels, preds, pos_label=1),
        "precision_irony": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_irony": recall_score(labels, preds, pos_label=1, zero_division=0),
        "recall_nonirony": recall_score(labels, preds, pos_label=0, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def normalize_text(s):
    """去 @提及 / #话题标签 / URL，压缩空白。

    用途：修复 hate 任务的分布偏移 —— test 的话题标签数是 train 的 3.2 倍
    （train 0.670 / val 0.722 / test 2.135），模型会把 # 当判别特征，
    导致 train 99.05% / val 78.48% / test 52.80% 的阶梯式崩塌。
    默认不启用，irony 与 FIGLANG 的行为保持完全不变。
    """
    s = re.sub(r"http\S+|www\.\S+", " ", str(s))
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_split(path, tokenizer, max_length, torch_format=False, normalize=False):
    """加载并分词一个数据划分。

    torch_format=True 时只保留模型输入列并转为张量，供 DataLoader 直接使用；
    Dataset.from_pandas 会带出 __index_level_0__ 等非张量列，不过滤会导致
    batch 中出现 list，进而在 v.to(device) 处抛 AttributeError。
    normalize=True 时先做文本归一化（见 normalize_text）。
    """
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    if normalize:
        df["text"] = df["text"].map(normalize_text)
        df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    ds = ds.map(
        lambda ex: tokenizer(ex["text"], padding="max_length",
                             truncation=True, max_length=max_length),
        batched=True, remove_columns=["text"],
    )
    if torch_format:
        keep = [c for c in ("input_ids", "attention_mask", "token_type_ids", "labels")
                if c in ds.column_names]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        ds.set_format(type="torch", columns=keep)
    return ds, df


def run_stage(stage_name, model_name, backbone, train_path, val_path, hp,
              init_from, out_dir, use_class_weight=True, seed=42,
              task_labels=("not ironic", "ironic"), normalize=False):
    # task_labels = (负类名, 正类名)。默认值即反讽任务，保持对现有调用的完全兼容。
    # hate 任务应传 ("not hateful", "hateful")。
    print("\n" + "=" * 92)
    print(f"【{stage_name}】骨干={backbone}  初始化自={init_from}")
    print(f"  训练集={train_path}  验证集={val_path}")
    print(f"  标签语义: 0={task_labels[0]}  1={task_labels[1]}")
    print(f"  epochs={hp['epochs']} lr={hp['lr']} bs={hp['bs']} "
          f"warmup_ratio={hp['warmup_ratio']} max_length={hp['max_length']}"
          f"  normalize_text={normalize}")
    print("=" * 92, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    train_ds, train_df = load_split(train_path, tokenizer, hp["max_length"],
                                    normalize=normalize)
    val_ds, val_df = load_split(val_path, tokenizer, hp["max_length"],
                                normalize=normalize)
    print(f"  训练 {len(train_ds):,} 条 / 验证 {len(val_ds):,} 条", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        init_from, num_labels=2,
        id2label={0: task_labels[0], 1: task_labels[1]},
        label2id={task_labels[0]: 0, task_labels[1]: 1},
    )

    cw, counts = compute_class_weights(train_df["label"].values)
    n_pos = int(counts[1])
    print(f"  类别分布: {task_labels[0]} {int(counts[0]):,} / {task_labels[1]} {n_pos:,}"
          f"  (正类 {n_pos/len(train_df)*100:.1f}%)")
    if use_class_weight:
        print(f"  类别权重: {cw.tolist()}  （逆频率，纠正召回不足）")
    else:
        cw = None
        print("  类别权重: 未启用（对照组）")

    steps_per_epoch = max(1, math.ceil(len(train_ds) / hp["bs"]))
    total_steps = steps_per_epoch * hp["epochs"]
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["bs"],
        per_device_eval_batch_size=hp["bs"] * 2,
        learning_rate=hp["lr"],
        warmup_ratio=hp["warmup_ratio"],
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir=f"./logs/{stage_name}_{model_name}",
        logging_steps=max(10, total_steps // 20),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_irony",   # 改用 F1，原用 accuracy 会偏向多数类
        greater_is_better=True,
        seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer_cls = WeightedTrainer if cw is not None else Trainer
    kwargs = dict(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=hp["patience"])])
    if cw is not None:
        kwargs["class_weights"] = cw
    trainer = trainer_cls(**kwargs)

    print(f"\n  总步数≈{total_steps}（每 epoch {steps_per_epoch} 步），开始训练…", flush=True)
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    ev = trainer.evaluate()
    print(f"\n  ★ {stage_name} 验证集结果:")
    for k in ["eval_accuracy", "eval_f1_irony", "eval_precision_irony",
              "eval_recall_irony", "eval_recall_nonirony"]:
        if k in ev:
            print(f"      {k:<26} {ev[k]:.4f}")
    print(f"      混淆矩阵 TN={ev.get('eval_tn')} FP={ev.get('eval_fp')} "
          f"FN={ev.get('eval_fn')} TP={ev.get('eval_tp')}")

    meta = {
        "stage": stage_name, "model_name": model_name, "backbone": backbone,
        "init_from": init_from, "train_path": train_path, "val_path": val_path,
        "hyperparams": hp, "class_weight": (cw.tolist() if cw is not None else None),
        "n_train": len(train_ds), "n_val": len(val_ds),
        "total_steps": total_steps, "eval": {k: (float(v) if isinstance(v, (int, float)) else v)
                                             for k, v in ev.items()},
        "best_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
    }
    with open(f"{out_dir}/train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"  已保存: {out_dir}", flush=True)
    return ev, meta


def evaluate_model(model_dir, name, eval_sets, max_length=128, normalize=False):
    """在多个测试集上评测，输出统一对照表。

    normalize 必须与训练时一致，否则训练用归一化文本、评测用原始文本，
    口径不一致会让 hate 这类分布偏移任务的评测结果失真。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    out = {"model": name, "path": model_dir, "device": device,
           "normalize_text": normalize, "max_length": max_length}
    print(f"\n{'─'*92}")
    print(f"评测【{name}】  device={device}  normalize={normalize}  "
          f"max_length={max_length}")
    for set_name, path in eval_sets.items():
        ds, df = load_split(path, tokenizer, max_length, torch_format=True,
                            normalize=normalize)
        dl = torch.utils.data.DataLoader(ds, batch_size=64)
        preds, probs = [], []
        with torch.no_grad():
            for b in dl:
                b = {k: v.to(device) for k, v in b.items()}
                logits = model(**b).logits
                p = torch.softmax(logits, dim=-1)
                probs.append(p.cpu().numpy())
                preds.append(p.argmax(-1).cpu().numpy())
        probs = np.concatenate(probs); preds = np.concatenate(preds)
        y = df["label"].values
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        acc = accuracy_score(y, preds)
        m = {
            "n": int(len(y)), "accuracy": round(acc, 4),
            "f1_irony": round(f1_score(y, preds, pos_label=1), 4),
            "precision_irony": round(precision_score(y, preds, pos_label=1, zero_division=0), 4),
            "recall_irony": round(recall_score(y, preds, pos_label=1, zero_division=0), 4),
            "recall_nonirony": round(recall_score(y, preds, pos_label=0, zero_division=0), 4),
            "pred_pro_rate": round(float((preds == 1).mean()), 4),
            "true_pro_rate": round(float((y == 1).mean()), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        out[set_name] = m
        print(f"  {set_name:<16} n={m['n']:<5} acc={acc*100:>6.2f}%  "
              f"F1(反讽)={m['f1_irony']:.4f}  召回(反讽)={m['recall_irony']*100:>6.2f}%  "
              f"召回(非反讽)={m['recall_nonirony']*100:>6.2f}%")
        # 保存置信度，供门控校准使用
        pd.DataFrame({
            "text": df["text"].values, "true_label": y, "predicted": preds,
            "conf_pro": probs[:, 1], "confidence": probs.max(axis=1),
        }).to_csv(f"experiments/results/conf_{name}_{set_name}.csv",
                  index=False, encoding="utf-8-sig")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="distilbert", choices=list(BACKBONES.keys()))
    ap.add_argument("--stage", default="all", choices=["1", "2", "all", "eval"])
    ap.add_argument("--no-class-weight", action="store_true",
                    help="对照组：不启用类别权重")
    ap.add_argument("--ablation-samedomain", action="store_true",
                    help="消融对照组 B：跳过 Misra 预训练，直接从原始骨干做 SemEval 微调，"
                         "但套用全部新训练策略。用于隔离「训练策略改进」与「跨域预训练」的贡献。")
    ap.add_argument("--eval-only-dir", default=None,
                    help="只评测指定模型目录，跳过训练")
    # ── 通用任务参数（默认值完全保持 irony/SemEval 行为不变，向后兼容）──
    ap.add_argument("--train-csv", default=SEMEVAL_TRAIN,
                    help="训练集 CSV（默认 SemEval irony_train）")
    ap.add_argument("--val-csv", default=SEMEVAL_VAL,
                    help="验证集 CSV（默认 SemEval irony_val）")
    ap.add_argument("--test-csv", default=None, action="append",
                    help="评测集 CSV，可重复传入；默认 irony_test + irony_val + misra_test")
    ap.add_argument("--out-name", default=None,
                    help="输出模型目录名（默认按 backbone/suffix/消融标志自动生成）")
    ap.add_argument("--task-labels", default="irony", choices=list(TASK_LABELS.keys()),
                    help="标签语义（决定 id2label 与日志类名）；默认 irony 保持兼容")
    ap.add_argument("--max-length", type=int, default=None,
                    help="截断长度（默认沿用 STAGE2_HP 的 128；FIGLANG 上下文版需 256）")
    ap.add_argument("--epochs", type=int, default=None,
                    help="微调轮数（默认 8；hate 实测 train 99.05% 严重过拟合，应减小）")
    ap.add_argument("--normalize-text", action="store_true",
                    help="训练与评测前归一化文本：去 @提及/#话题/URL、压缩空白。"
                         "用于修复 hate 的话题标签分布偏移（test 的 # 数是 train 的 3.2 倍）")
    args = ap.parse_args()

    os.makedirs("experiments/results", exist_ok=True)
    bb = BACKBONES[args.backbone]
    suffix = "cw" if not args.no_class_weight else "nocw"
    if args.out_name:
        # 通用任务模式：目录名由调用方指定，不与 irony 的两阶段/消融目录冲突
        s1_dir = None
        s2_dir = f"./models/{args.out_name}"
    elif args.ablation_samedomain:
        # 独立目录，避免覆盖两阶段模型
        s1_dir = None
        s2_dir = f"./models/judge_{args.backbone}_{suffix}_samedomain_ablation"
    else:
        s1_dir = f"./models/stage1_{args.backbone}_{suffix}"
        s2_dir = f"./models/judge_{args.backbone}_{suffix}_2stage"

    if args.test_csv:
        eval_sets = {os.path.splitext(os.path.basename(p))[0]: p
                     for p in args.test_csv}
    else:
        eval_sets = {
            "semeval_test": SEMEVAL_TEST,   # 主评测（与现有 n=150/n=200 可比）
            "semeval_val": SEMEVAL_VAL,
            "misra_test": MISRA_TEST,       # 跨域诊断
        }

    if args.eval_only_dir:
        r = evaluate_model(args.eval_only_dir, os.path.basename(args.eval_only_dir),
                           eval_sets, max_length=args.max_length or 128,
                           normalize=args.normalize_text)
        print("\n" + json.dumps(r, ensure_ascii=False, indent=2))
        return

    results = {}

    # 超参覆盖：--max-length / --epochs 未指定时完全沿用 STAGE2_HP（irony 行为不变）
    hp = dict(STAGE2_HP)
    if args.max_length is not None:
        hp["max_length"] = args.max_length
    if args.epochs is not None:
        hp["epochs"] = args.epochs
    # 通用任务模式下用 --train-csv/--val-csv；irony 模式下默认值即 SemEval，行为不变
    train_csv, val_csv = args.train_csv, args.val_csv

    if args.ablation_samedomain:
        # 消融对照组 B：不做 Misra 预训练，直接从原始骨干微调。
        # 与两阶段模型共用全部新训练策略，故 C−B 即跨域预训练的净贡献。
        is_irony = (train_csv == SEMEVAL_TRAIN)
        print("\n" + "=" * 92)
        print("【消融对照组 B】同域新策略：跳过 Misra 预训练")
        if is_irony:
            print("  目的: 隔离「训练策略改进」与「跨域预训练」各自的贡献")
            print("  对照: A 原基线(旧策略,无预训练) / B 本组(新策略,无预训练) / C 两阶段(新策略+预训练)")
            print("        B−A = 训练策略净贡献    C−B = 跨域预训练净贡献")
        else:
            print(f"  通用任务模式: train={train_csv}  val={val_csv}")
        print("=" * 92, flush=True)
        ev2, _ = run_stage("消融B-同域新策略微调", os.path.basename(s2_dir),
                           bb, train_csv, val_csv, hp,
                           init_from=bb, out_dir=s2_dir,
                           use_class_weight=not args.no_class_weight,
                           task_labels=TASK_LABELS[args.task_labels],
                           normalize=args.normalize_text)
        results["ablation_samedomain_val"] = ev2
        results["ablation"] = {"kind": "samedomain_new_strategy",
                               "pretrain": "none", "init_from": bb}
        final_dir = s2_dir
    else:
        if args.stage in ("1", "all"):
            ev1, _ = run_stage("阶段1-域适应预训练", f"stage1_{args.backbone}_{suffix}",
                               bb, MISRA_TRAIN, MISRA_VAL, STAGE1_HP,
                               init_from=bb, out_dir=s1_dir,
                               use_class_weight=not args.no_class_weight)
            results["stage1_val"] = ev1

        if args.stage in ("2", "all"):
            init = s1_dir if args.stage == "all" or os.path.isdir(s1_dir) else bb
            if not os.path.isdir(init):
                print(f"[WARN] 阶段1 输出不存在({init})，阶段2 回退到原始骨干 {bb}")
                init = bb
            ev2, _ = run_stage("阶段2-任务微调", f"judge_{args.backbone}_{suffix}_2stage",
                               bb, SEMEVAL_TRAIN, SEMEVAL_VAL, STAGE2_HP,
                               init_from=init, out_dir=s2_dir,
                               use_class_weight=not args.no_class_weight)
            results["stage2_val"] = ev2
        final_dir = s2_dir if args.stage in ("2", "all") else s1_dir

    print("\n\n" + "=" * 92)
    print("最终评测（多测试集）")
    print("=" * 92)
    # 通用任务模式（指定了 --out-name）用该名称做 tag，避免覆盖 irony 的结果文件
    if args.out_name:
        tag = args.out_name
    elif args.ablation_samedomain:
        tag = f"{args.backbone}_{suffix}_samedomain"
    else:
        tag = f"{args.backbone}_{suffix}"
    if os.path.isdir(final_dir):
        r = evaluate_model(final_dir, os.path.basename(final_dir), eval_sets,
                           max_length=args.max_length or 128,
                           normalize=args.normalize_text)
        results["final"] = r
        with open(f"experiments/results/retrain_{tag}_result.json",
                  "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n结果已写入 experiments/results/retrain_{tag}_result.json")
    else:
        print(f"[WARN] 未找到模型目录 {final_dir}")


if __name__ == "__main__":
    main()
