"""准备 FIGLANG 2020 上下文讽刺数据集（泛化性实验）。

为什么选它（四项筛选标准全部满足，见 probe_tweeteval_figlang.py）：
  ① 困难    ：多数类基线恰为 50.0%，无捷径可走（对照 SST-2 裁决者 98% 已饱和）
  ② 同族    ：同为讽刺检测，视角分化辩论的机理可迁移
  ③ 二分类  ：票数门槛、偏置分析可直接复用（label 为 SARCASM / NOT_SARCASM 字符串，需映射）
  ④ 含上下文：这是与当前 irony 任务【本质不同】的难度来源——
              单句 irony 只需看文本本身，FIGLANG 需结合 context 判断 response 是否讽刺

⚠️ 标注机制核验（避免重蹈 Misra 覆辙）：
   Misra 按【新闻源】标注（TheOnion=讽刺），模型可靠识别文风作弊，
   导致域内 +28.33pp 但跨域迁移效率仅 4.5%。
   FIGLANG 2020 是学术共享任务，标签为人工标注的 SARCASM/NOT_SARCASM，
   非按来源推导 → 无文风捷径。本脚本额外核验标签是否与 subreddit 等来源字段相关。

数据格式：jsonl，字段 label / context / response（training 无 id，testing 有 id）
输出：与 irony_*.csv 一致的 text,label 两列，其中 text = context + response 拼接。
      另存原始三列版本 figlang_*_raw.csv 供「仅 response」的消融对照。
"""
import os
import io
import json
import sys
import urllib.request

import pandas as pd

BASE = "https://hf-mirror.com/datasets/tasksource/figlang2020-sarcasm/resolve/main"
OUT_DIR = "data/raw"
RES = "experiments/results"

FILES = {
    "twitter_train": "sarcasm_detection_shared_task_twitter_training.jsonl",
    "twitter_test": "sarcasm_detection_shared_task_twitter_testing.jsonl",
    "reddit_train": "sarcasm_detection_shared_task_reddit_training.jsonl",
    "reddit_test": "sarcasm_detection_shared_task_reddit_testing.jsonl",
}
LABEL_MAP = {"SARCASM": 1, "NOT_SARCASM": 0}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def fetch(url, timeout=240):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def join_ctx_resp(context, response, max_ctx_chars=600):
    """拼接上下文与回复。

    context 在原始数据中是被截断的 token 流（无标点、常为小写），
    过长会挤占 1.5B 的有效注意力，故截断到 max_ctx_chars。
    """
    ctx = str(context).strip()
    resp = str(response).strip()
    if not ctx:
        return resp
    if len(ctx) > max_ctx_chars:
        ctx = ctx[-max_ctx_chars:]
    return f"Context: {ctx}\nResponse: {resp}"


print("=" * 104)
print("准备 FIGLANG 2020 上下文讽刺数据集")
print("=" * 104)

profile = {}
for split, rel in FILES.items():
    url = f"{BASE}/{rel}"
    try:
        raw = fetch(url)
    except Exception as e:
        print(f"  [FAIL] {split}: {type(e).__name__}: {e}")
        sys.exit(1)
    try:
        df = pd.read_json(io.BytesIO(raw), lines=True)
    except Exception as e:
        print(f"  [解析失败] {split}: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"\n{'─'*104}")
    print(f"◆ {split}   原始 {len(df):,} 条   {len(raw)/1024/1024:.2f} MB")
    print(f"  列: {list(df.columns)}")

    # 标签映射与校验
    if "label" not in df.columns:
        print(f"  [FAIL] 缺少 label 列")
        sys.exit(1)
    unknown = set(df["label"].unique()) - set(LABEL_MAP)
    if unknown:
        print(f"  [FAIL] 存在未知标签值: {unknown}")
        sys.exit(1)
    df["label"] = df["label"].map(LABEL_MAP).astype(int)
    df = df.dropna(subset=["label", "response"])
    df["response"] = df["response"].astype(str).str.strip()
    df["context"] = df["context"].astype(str).str.strip()
    df = df[df["response"].str.len() > 0].reset_index(drop=True)

    vc = df["label"].value_counts().sort_index()
    print(f"  清洗后 {len(df):,} 条   标签分布: "
          f"{ {int(k): int(v) for k, v in vc.items()} }")
    print(f"  正类(SARCASM)占比 {(df.label==1).mean()*100:.2f}%   "
          f"多数类基线 {vc.max()/len(df)*100:.2f}%")

    # 文本长度
    rl = df["response"].str.split().str.len()
    cl = df["context"].str.split().str.len()
    print(f"  response 词数: 均值 {rl.mean():.1f} 中位 {rl.median():.0f} 最大 {rl.max()}")
    print(f"  context  词数: 均值 {cl.mean():.1f} 中位 {cl.median():.0f} 最大 {cl.max()}")

    # ★ 标注机制核验：标签是否与来源字段相关（Misra 的作弊路径）
    print(f"  ◆ 标注机制核验（是否存在来源捷径）")
    leak_risk = []
    for col in df.columns:
        if col in ("label", "context", "response"):
            continue
        try:
            nun = df[col].nunique()
            if 1 < nun <= 20:
                ct = pd.crosstab(df[col], df.label, normalize="index")
                if ct.shape[1] == 2:
                    spread = (ct[1].max() - ct[1].min()) * 100
                    flag = "高风险!" if spread > 25 else "低风险"
                    print(f"    [{col}] {nun} 个取值，正类率极差 {spread:.1f}pp  → {flag}")
                    if spread > 25:
                        leak_risk.append({"column": col, "spread_pp": round(spread, 2),
                                          "rate_by_value": ct[1].round(4).to_dict()})
            elif nun > 20:
                print(f"    [{col}] {nun} 个唯一值（高基数，非来源字段）")
        except Exception:
            pass
    if not leak_risk:
        print(f"    → 未发现来源捷径，标注机制可靠（不同于 Misra 按新闻源标注）")

    # 样例
    print(f"  样例（label=1 SARCASM）:")
    for _, r in df[df.label == 1].head(2).iterrows():
        print(f"    ctx: {r['context'][:88]}")
        print(f"    rsp: {r['response'][:88]}")
    print(f"  样例（label=0 NOT_SARCASM）:")
    for _, r in df[df.label == 0].head(2).iterrows():
        print(f"    ctx: {r['context'][:88]}")
        print(f"    rsp: {r['response'][:88]}")

    # ── 落盘：拼接版（主）+ 原始三列版（消融对照用）──
    df["text"] = [join_ctx_resp(c, r) for c, r in zip(df["context"], df["response"])]
    out_main = f"{OUT_DIR}/figlang_{split}.csv"
    df[["text", "label"]].to_csv(out_main, index=False, encoding="utf-8")
    out_raw = f"{OUT_DIR}/figlang_{split}_raw.csv"
    keep = [c for c in ["context", "response", "label", "id"] if c in df.columns]
    df[keep].to_csv(out_raw, index=False, encoding="utf-8")

    # 仅 response 版（消融：上下文是否真的带来难度）
    out_resp = f"{OUT_DIR}/figlang_{split}_responseonly.csv"
    df[["response", "label"]].rename(columns={"response": "text"}) \
        .to_csv(out_resp, index=False, encoding="utf-8")

    joined_wl = df["text"].str.split().str.len()
    print(f"  → {out_main}  ({len(df):,} 条, 拼接后词数均值 {joined_wl.mean():.1f} "
          f"最大 {joined_wl.max()})")
    print(f"  → {out_raw}  (原始三列)")
    print(f"  → {out_resp}  (仅 response，供上下文消融对照)")

    profile[split] = {
        "rows": len(df), "pos_rate_pct": round((df.label == 1).mean() * 100, 2),
        "majority_baseline_pct": round(vc.max() / len(df) * 100, 2),
        "response_words_mean": round(float(rl.mean()), 1),
        "context_words_mean": round(float(cl.mean()), 1),
        "joined_words_mean": round(float(joined_wl.mean()), 1),
        "joined_words_max": int(joined_wl.max()),
        "columns": list(df.columns),
        "label_leak_risk": leak_risk,
        "files": {"main": out_main, "raw": out_raw, "response_only": out_resp},
    }

# ── 泄漏检查 ──
print(f"\n{'─'*104}")
print("泄漏检查")
print("─" * 104)
for dom in ["twitter", "reddit"]:
    tr = f"{OUT_DIR}/figlang_{dom}_train.csv"
    te = f"{OUT_DIR}/figlang_{dom}_test.csv"
    if os.path.exists(tr) and os.path.exists(te):
        a = set(pd.read_csv(tr)["text"].astype(str).str.strip())
        b = set(pd.read_csv(te)["text"].astype(str).str.strip())
        n = len(a & b)
        print(f"  figlang_{dom}: train ∩ test = {n}  [{'OK' if n == 0 else '泄漏!'}]")
        # response 层面（拼接前的原始回复）
        ra = set(pd.read_csv(f"{OUT_DIR}/figlang_{dom}_train_responseonly.csv")["text"].astype(str).str.strip())
        rb = set(pd.read_csv(f"{OUT_DIR}/figlang_{dom}_test_responseonly.csv")["text"].astype(str).str.strip())
        nr = len(ra & rb)
        print(f"  figlang_{dom}: response 层面 train ∩ test = {nr}  "
              f"[{'OK' if nr == 0 else '需注意：上下文可区分但回复重复'}]")
# 跨域
for a, b in [("figlang_twitter_train", "figlang_reddit_train"),
             ("figlang_twitter_test", "figlang_reddit_test")]:
    pa, pb = f"{OUT_DIR}/{a}.csv", f"{OUT_DIR}/{b}.csv"
    if os.path.exists(pa) and os.path.exists(pb):
        sa = set(pd.read_csv(pa)["text"].astype(str).str.strip())
        sb = set(pd.read_csv(pb)["text"].astype(str).str.strip())
        print(f"  {a} ∩ {b} = {len(sa & sb)}")
# 与现有 irony / hate 的交叉
for other in ["irony_test", "hate_test"]:
    po = f"{OUT_DIR}/{other}.csv"
    if not os.path.exists(po):
        continue
    so = set(pd.read_csv(po)["text"].astype(str).str.strip())
    for split in profile:
        p = profile[split]["files"]["main"]
        s = set(pd.read_csv(p)["text"].astype(str).str.strip())
        n = len(so & s)
        if n:
            print(f"  {other} ∩ figlang_{split} = {n}  [需检查]")
print("  （figlang text 含 'Context:'/'Response:' 前缀，与 irony/hate 纯文本不会重叠）")

# ── 任务总览 ──
print(f"\n{'='*104}")
print("任务总览（三个任务的难度与结构对比）")
print("=" * 104)
print(f"  {'任务':<26} {'训练':>8} {'测试':>8} {'正类占比':>10} {'多数类基线':>11} "
      f"{'词数均值':>9} {'含上下文':>9}")
print("  " + "-" * 92)
rows = []
for task, tr_p, te_p, has_ctx in [
    ("tweet_eval/irony（当前）", "data/raw/irony_train.csv", "data/raw/irony_test.csv", False),
    ("tweet_eval/hate", "data/raw/hate_train.csv", "data/raw/hate_test.csv", False),
    ("FIGLANG twitter", "data/raw/figlang_twitter_train.csv", "data/raw/figlang_twitter_test.csv", True),
    ("FIGLANG reddit", "data/raw/figlang_reddit_train.csv", "data/raw/figlang_reddit_test.csv", True),
]:
    if not (os.path.exists(tr_p) and os.path.exists(te_p)):
        continue
    tr = pd.read_csv(tr_p)
    te = pd.read_csv(te_p)
    pos = (te.label == 1).mean() * 100
    mb = te["label"].value_counts().max() / len(te) * 100
    wl = te["text"].astype(str).str.split().str.len().mean()
    rows.append({"task": task, "train_rows": len(tr), "test_rows": len(te),
                 "test_pos_rate_pct": round(pos, 2),
                 "majority_baseline_pct": round(mb, 2),
                 "words_mean": round(float(wl), 1), "has_context": has_ctx})
    print(f"  {task:<26} {len(tr):>8,} {len(te):>8,} {pos:>9.2f}% {mb:>10.2f}% "
          f"{wl:>9.1f} {'是' if has_ctx else '否':>9}")

with open(f"{RES}/figlang_dataset_profile.json", "w", encoding="utf-8") as f:
    json.dump({"figlang": profile, "all_tasks": rows,
               "label_semantics": {"SARCASM": "1 → pro", "NOT_SARCASM": "0 → con"},
               "why_chosen": ["多数类基线 50% 无捷径", "与 irony 同族（讽刺检测）",
                              "二分类可复用票数门槛与偏置分析",
                              "含 context，难度来源与单句 irony 不同",
                              "人工标注非来源推导，无 Misra 式文风捷径"],
               "ablation_available": ("每 split 均存三份：拼接版(主)、原始三列版、"
                                      "仅response版 —— 可量化「上下文」本身带来的难度")},
              f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已写入 {RES}/figlang_dataset_profile.json")
print("=" * 104)
