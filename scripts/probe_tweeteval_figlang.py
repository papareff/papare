"""聚焦核实两个最可能的困难任务来源：tweet_eval 子集 与 FIGLANG 2020 jsonl。

前一轮发现：
  • SetFit/implicit_hate_v1、hate_speech_offensive 均不可用（隐性仇恨候选落空）
  • sarcasm-2.0-reddit 测试集标签 100% UNKNOWN → 无法评测，排除
  • MUSTARD 只有 test 200 条且含音频 → 规模不足，排除
  • friends_chandler_bing 无测试集、其余分片为对话微调格式 → 排除
剩余候选：tweet_eval（最权威，含 hate/offensive/irony/stance 等子集）
          FIGLANG 2020（学术共享任务，jsonl，标注质量高）

核实要点：
  ① tweet_eval 有哪些子集、各自规模与标签分布
  ② 哪个子集是「困难且与反讽同族」——hate / offensive 属隐性攻击，与反讽的
     「字面≠意图」结构同构；stance 属立场检测，需背景知识，也困难
  ③ FIGLANG jsonl 的真实规模与字段（是否含上下文）
"""
import urllib.request
import json
import io
import os

BASE = "https://hf-mirror.com"

import pyarrow.parquet as pq
import pandas as pd


def fetch(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parquet_profile(url):
    """完整下载小 parquet 并给出规模/标签/长度画像；大文件只读 footer 取行数。"""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
        size_mb = int(cl) / 1024 / 1024 if cl else None
    except Exception:
        size_mb = None

    if size_mb and size_mb > 40:
        return {"size_mb": round(size_mb, 2), "rows": None,
                "note": "文件过大，仅记录体积"}

    raw = fetch(url)
    df = pq.read_table(io.BytesIO(raw)).to_pandas()
    out = {"size_mb": round(len(raw) / 1024 / 1024, 2), "rows": len(df),
           "columns": list(df.columns)}
    lab = next((c for c in df.columns if str(c).lower() == "label"), None)
    if lab is not None:
        vc = df[lab].value_counts(dropna=False).sort_index()
        out["label_dist"] = {str(k): int(v) for k, v in vc.items()}
        out["n_classes"] = int(df[lab].nunique())
        # 多数类基线：低于它说明模型没学到东西；接近 100% 说明是饱和/不平衡任务
        out["majority_baseline_pct"] = round(vc.max() / len(df) * 100, 2)
    txt = next((c for c in df.columns
                if str(c).lower() in ("text", "sentence", "tweet", "content")), None)
    if txt is not None:
        wl = df[txt].astype(str).str.split().str.len()
        out["words_mean"] = round(float(wl.mean()), 1)
        out["words_max"] = int(wl.max())
        out["samples"] = [str(t)[:110] for t in df[txt].head(3)]
    return out


print("=" * 108)
print("【1】tweet_eval 子集清单与规模")
print("=" * 108)

url = f"{BASE}/api/datasets/tweet_eval"
info = json.loads(fetch(url).decode("utf-8", "replace"))
files = [s.get("rfilename") for s in (info.get("siblings") or [])]
print(f"  总文件数 {len(files)}")

# 从文件名推断子集
subsets = {}
for f in files:
    f = str(f)
    if not f.endswith(".parquet"):
        continue
    parts = f.split("/")
    if len(parts) >= 2:
        subsets.setdefault(parts[0], []).append(f)
print(f"  子集数 {len(subsets)}: {', '.join(sorted(subsets))}")

TE_PROFILES = {}
for sub in sorted(subsets):
    fs = subsets[sub]
    print(f"\n{'─'*108}")
    print(f"  ◆ 子集: {sub}   ({len(fs)} 个文件)")
    for split_kind in ["train", "test", "validation"]:
        rel = next((f for f in fs if split_kind in str(f)), None)
        if rel is None:
            continue
        u = f"{BASE}/datasets/tweet_eval/resolve/main/{rel}"
        try:
            p = parquet_profile(u)
        except Exception as e:
            print(f"    [FAIL] {split_kind}: {type(e).__name__}: {str(e)[:70]}")
            continue
        TE_PROFILES[f"{sub}/{split_kind}"] = p
        print(f"    {split_kind:<11} 行数 {str(p.get('rows')):>7}  "
              f"大小 {p.get('size_mb')} MB  类别数 {p.get('n_classes', '—')}")
        if p.get("label_dist"):
            print(f"                  标签分布 {p['label_dist']}")
            print(f"                  多数类基线 {p.get('majority_baseline_pct')}%")
        if split_kind == "test" and p.get("samples"):
            print(f"                  词数均值 {p.get('words_mean')} 最大 {p.get('words_max')}")
            for s in p["samples"]:
                print(f"                    • {s}")

print(f"\n\n{'=' * 108}")
print("【2】FIGLANG 2020 讽刺（jsonl，学术共享任务）")
print("=" * 108)
FIG = [
    ("twitter_training", "sarcasm_detection_shared_task_twitter_training.jsonl"),
    ("twitter_testing", "sarcasm_detection_shared_task_twitter_testing.jsonl"),
    ("reddit_training", "sarcasm_detection_shared_task_reddit_training.jsonl"),
    ("reddit_testing", "sarcasm_detection_shared_task_reddit_testing.jsonl"),
]
FIG_PROFILES = {}
for name, rel in FIG:
    u = f"{BASE}/datasets/tasksource/figlang2020-sarcasm/resolve/main/{rel}"
    try:
        raw = fetch(u, timeout=240)
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:70]}")
        continue
    try:
        df = pd.read_json(io.BytesIO(raw), lines=True)
    except Exception as e:
        print(f"  [解析失败] {name}: {type(e).__name__}: {str(e)[:70]}")
        continue
    print(f"\n{'─'*108}")
    print(f"  ◆ {name}   行数 {len(df):,}   大小 {len(raw)/1024/1024:.2f} MB")
    print(f"    列 {list(df.columns)}")
    lab = next((c for c in df.columns
                if str(c).lower() in ("label", "sarcastic", "class", "y", "gold")), None)
    if lab is not None:
        vc = df[lab].value_counts(dropna=False)
        print(f"    标签列 [{lab}] 共 {df[lab].nunique()} 类:")
        for k, v in vc.items():
            print(f"        {str(k)[:40]:<42} {v:>7,} ({v/len(df)*100:>5.1f}%)")
        print(f"    多数类基线 {vc.max()/len(df)*100:.2f}%")
    txt = next((c for c in df.columns
                if str(c).lower() in ("text", "tweet", "comment", "content",
                                      "sentence", "body")), None)
    if txt is not None:
        wl = df[txt].astype(str).str.split().str.len()
        print(f"    文本列 [{txt}] 词数: 均值 {wl.mean():.1f} 中位 {wl.median():.0f} "
              f"最大 {wl.max()}")
        for t in df[txt].astype(str).head(3):
            print(f"        • {t[:120]}")
    ctx = [c for c in df.columns if str(c).lower() in
           ("context", "parent", "conversation", "history", "response")]
    print(f"    上下文字段: {ctx if ctx else '无（单句任务）'}")
    FIG_PROFILES[name] = {"rows": len(df), "columns": list(df.columns),
                          "size_mb": round(len(raw) / 1024 / 1024, 2)}

# ── 难度与同构性评估 ──
print(f"\n\n{'=' * 108}")
print("【3】候选评估：困难度 + 与反讽的同构性 + 可用性")
print("=" * 108)
print(f"  {'候选':<34} {'测试集规模':>11} {'多数类基线':>11} {'类别数':>7} "
      f"{'含上下文':>9} {'可用性':>9}")
print("  " + "-" * 96)

EVAL = []
for sub in sorted(subsets):
    k = f"{sub}/test"
    p = TE_PROFILES.get(k)
    if not p or p.get("rows") is None:
        continue
    mb = p.get("majority_baseline_pct")
    # 困难度判据：多数类基线越接近 50%（均衡）且任务本身语义复杂 → 越难
    # 饱和判据：多数类基线 > 70% 且属情感/表情类 → 可能饱和
    hard = "困难" if mb is not None and mb < 65 else ("中等" if mb and mb < 80 else "易饱和?")
    EVAL.append({"id": f"tweet_eval/{sub}", "test_rows": p["rows"],
                 "majority_baseline_pct": mb, "n_classes": p.get("n_classes"),
                 "has_context": False, "difficulty": hard, "available": True})
    print(f"  tweet_eval/{sub:<22} {p['rows']:>11,} {str(mb)+'%':>11} "
          f"{str(p.get('n_classes')):>7} {'否':>9} {'可用':>9}  [{hard}]")

for name in ["twitter_testing", "reddit_testing"]:
    p = FIG_PROFILES.get(name)
    if not p:
        continue
    EVAL.append({"id": f"figlang2020/{name}", "test_rows": p["rows"],
                 "majority_baseline_pct": None, "n_classes": None,
                 "has_context": False, "difficulty": "困难", "available": True})
    print(f"  figlang2020/{name:<22} {p['rows']:>11,} {'—':>11} {'—':>7} "
          f"{'否':>9} {'可用':>9}  [困难]")

print(f"\n  ◆ 已排除（前一轮核实）")
print(f"    sarcasm-2.0-reddit : 测试集标签 100% UNKNOWN → 无法评测")
print(f"    MUSTARD            : 仅 test 200 条且含音频 → 规模不足、多模态")
print(f"    friends_chandler   : 无测试集，其余分片为对话微调格式 → 不可评测")
print(f"    SetFit/implicit_hate_v1、hate_speech_offensive : 仓库不可用（401）")

with open("experiments/results/hard_task_shortlist.json", "w",
          encoding="utf-8") as f:
    json.dump({"tweet_eval_profiles": TE_PROFILES,
               "figlang_profiles": FIG_PROFILES,
               "shortlist": EVAL}, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已写入 experiments/results/hard_task_shortlist.json")
print("=" * 108)
