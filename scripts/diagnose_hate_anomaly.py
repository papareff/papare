"""诊断 hate 裁决者在 test 上 pred_pro_rate 86.78% 而 val 仅 47.85% 的异常。
同一模型在输入分布相似的两个集合上预测率差 39pp，需逐层排查：
L1 数据层：CSV 真实行数/标签分布/文本长度/编码/列名
L2 加载层：datasets 缓存指纹是否与其他文件冲突（load_split 默认启用 mmap 缓存）
L3 推理层：用独立加载路径（pandas + tokenizer 手工批处理）重算预测率，与训练脚本结果对照
L4 模型层：直接读 checkpoint 的 logits 分布，看是否退化成常数输出
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MODEL = "./models/judge_hate_distilbert_cw_samedomain"
SPLITS = {
    "hate_val": "data/raw/hate_val.csv",
    "hate_test": "data/raw/hate_test.csv",
    "hate_train": "data/raw/hate_train.csv",
}
print("=" * 104)
print("hate 裁决者 test 异常诊断")
print("=" * 104)

# ── L1 数据层 ──
print("\n【L1】CSV 真实内容")
frames = {}
for sp, p in SPLITS.items():
    df = pd.read_csv(p)
    frames[sp] = df
    wc = df["text"].astype(str).str.split().str.len()
    print(f"  {sp:<11} 行数 {len(df):>5,}  列 {list(df.columns)}")
    print(f"              label 分布 {df['label'].value_counts().to_dict()}  "
          f"正类占比 {df['label'].mean()*100:.2f}%")
    print(f"              文本词数 min={wc.min()} mean={wc.mean():.1f} "
          f"max={wc.max()}  空文本 {int((wc == 0).sum())}")
    print(f"              重复文本 {int(df['text'].duplicated().sum())} 条")

# val 与 test 的长度/标签分布差异
wv = frames["hate_val"]["text"].astype(str).str.split().str.len()
wt = frames["hate_test"]["text"].astype(str).str.split().str.len()
print(f"\n  val vs test 词数均值差: {wv.mean()-wt.mean():+.2f}")
print(f"  val vs test 正类占比差: "
      f"{frames['hate_val']['label'].mean()-frames['hate_test']['label'].mean():+.4f}")

# ── L3 独立推理（不复用 load_split，绕开 datasets 缓存）──
print("\n【L3】独立推理路径重算（pandas + 手工分批，绕开 datasets mmap 缓存）")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev).eval()
print(f"  device={dev}  id2label={model.config.id2label}")

indep = {}
for sp, df in frames.items():
    if sp == "hate_train":
        df = df.sample(n=2000, random_state=42)
    texts = df["text"].astype(str).tolist()
    y = df["label"].values
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i+64], truncation=True, padding=True,
                      max_length=128, return_tensors="pt").to(dev)
            lg = model(**enc).logits
            p = torch.softmax(lg, dim=-1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(-1).cpu().numpy())
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    acc = accuracy_score(y, preds)
    indep[sp] = {
        "n": int(len(y)), "acc": float(acc),
        "pred_pro_rate": float(preds.mean()), "true_pro_rate": float(y.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "mean_p1": float(probs[:, 1].mean()),
        "logit_std": float(probs[:, 1].std()),
    }
    print(f"  {sp:<11} n={len(y):>5,} acc={acc*100:.2f}%  "
          f"pred_pro={preds.mean()*100:.2f}%  true_pro={y.mean()*100:.2f}%  "
          f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"              p1 均值={probs[:,1].mean():.4f}  "
          f"p1 标准差={probs[:,1].std():.4f}  "
          f"p1∈(0.4,0.6) 占比={float(((probs[:,1]>0.4)&(probs[:,1]<0.6)).mean())*100:.2f}%")

# ── 对照训练脚本的官方评测结果 ──
print("\n【对照】训练脚本 evaluate_model 的官方结果")
try:
    with open("experiments/results/retrain_hate_distilbert_cw_result.json",
              encoding="utf-8") as f:
        off = json.load(f)["final"]
    for k in ["hate_val", "hate_test"]:
        o = off[k]
        i = indep["hate_val" if k == "hate_val" else "hate_test"]
        print(f"  {k:<10} 官方 acc={o['accuracy']*100:.2f}% pred_pro={o['pred_pro_rate']*100:.2f}%"
              f"  |  独立 acc={i['acc']*100:.2f}% pred_pro={i['pred_pro_rate']*100:.2f}%")
        d_acc = abs(o["accuracy"] - i["acc"]) * 100
        d_pro = abs(o["pred_pro_rate"] - i["pred_pro_rate"]) * 100
        flag = "✗ 不一致" if (d_acc > 1.0 or d_pro > 1.0) else "✓ 一致"
        print(f"              差异 acc {d_acc:.2f}pp / pred_pro {d_pro:.2f}pp  {flag}")
except Exception as e:
    print(f"  读取官方结果失败: {e}")

# ── L2 datasets 缓存指纹检查 ──
print("\n【L2】datasets 缓存指纹检查")
try:
    from datasets import load_dataset, disable_caching
    fps = {}
    for sp, p in SPLITS.items():
        ds = load_dataset("csv", data_files=p, split="train")
        fps[sp] = ds._fingerprint
        print(f"  {sp:<11} fingerprint={ds._fingerprint}  len={len(ds)}")
    if len(set(fps.values())) < len(fps):
        print("  ✗ 存在指纹冲突！datasets 会返回错误缓存内容")
    else:
        print("  ✓ 指纹互不相同")
except Exception as e:
    print(f"  检查失败: {type(e).__name__}: {e}")

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/hate_anomaly_diagnosis.json", "w",
          encoding="utf-8") as f:
    json.dump({"independent_inference": indep}, f, ensure_ascii=False, indent=2)
print("\n" + "=" * 104)
print("结果已写入 experiments/results/hate_anomaly_diagnosis.json")
