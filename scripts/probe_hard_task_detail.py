"""核查困难任务候选的真实规模、标签体系、是否含上下文、难度证据。

防重蹈 Misra 覆辙：Misra 按【新闻源】标注（TheOnion=讽刺），模型可靠识别文风作弊，
导致域内 +28pp 但跨域迁移效率仅 4.5%。故此处必须核验：
  ① 标注机制是否依赖表层可作弊特征
  ② 是否真含上下文（contextual sarcasm 的难度来源）
  ③ 类别比例与测试集规模（决定跑全量的算力成本）
"""
import urllib.request
import json
import io
import os

BASE = "https://hf-mirror.com"
TMP = "data/tmp_hardtask"
os.makedirs(TMP, exist_ok=True)

# SetFit 系列 + 其他隐性仇恨/上下文讽刺候选
IDS = [
    "SetFit/implicit_hate_v1",
    "SetFit/imdb",
    "SetFit/subj",
    "SetFit/ag_news",
    "SetFit/20_newsgroups",
    "SetFit/SST-5",
    "SetFit/CR",
    "SetFit/MPQA",
    "SetFit/TREC-QC",
    "michellemwange25/hate_speech_offensive",
    "dair-ai/emotion",
    "tasksource/figlang2020-sarcasm",
    "siddharthyadev/sarcasm-2.0-reddit",
    "CreativeLang/SARC_Sarcasm",
    "macabdul9/SarcasmDetection_Mustard",
    "yl2342/friends_chandler_bing_sarcasm",
    "tweet_eval",
]


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def probe_exists(ds_id):
    """依次尝试 datasets / models 两种仓库类型，返回 (kind, info)。"""
    for kind in ["datasets", "models"]:
        try:
            info = json.loads(get(f"{BASE}/api/{kind}/{ds_id}"))
            return kind, info
        except Exception:
            continue
    return None, None


print("=" * 106)
print("困难任务候选可用性核查")
print("=" * 106)

found = []
for ds_id in IDS:
    kind, info = probe_exists(ds_id)
    if kind is None:
        print(f"  [不可用] {ds_id}")
        continue
    sib = info.get("siblings") or []
    files = [s.get("rfilename") for s in sib]
    n_pq = sum(1 for f in files if str(f).endswith(".parquet"))
    n_jsonl = sum(1 for f in files if str(f).endswith((".jsonl", ".json")))
    found.append({
        "id": ds_id, "kind": kind,
        "downloads": info.get("downloads", 0),
        "likes": info.get("likes", 0),
        "n_files": len(files), "n_parquet": n_pq, "n_jsonl": n_jsonl,
        "files": files[:14],
        "tags": [t for t in info.get("tags", [])
                 if not t.startswith(("region", "size_categories"))][:12],
    })
    print(f"\n  [OK {kind}] {ds_id}   下载量={info.get('downloads')} "
          f"点赞={info.get('likes')}  文件={len(files)} "
          f"(parquet {n_pq} / jsonl {n_jsonl})")
    print(f"      文件: {', '.join(str(f) for f in files[:8])}")
    tags = [t for t in info.get("tags", [])
            if not t.startswith(("region", "size_categories"))]
    if tags:
        print(f"      标签: {', '.join(tags[:10])}")

with open("experiments/results/hard_task_availability.json", "w",
          encoding="utf-8") as f:
    json.dump(found, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 106)
print(f"可用候选 {len(found)}/{len(IDS)}")
print("=" * 106)

# ── 下载小规模候选，核实真实规模与标签体系 ──
print("\n" + "=" * 106)
print("下载核实（仅小文件；大文件只读 parquet footer 取行数）")
print("=" * 106)

try:
    import pyarrow.parquet as pq
    import pandas as pd
    HAS_PA = True
except ImportError:
    HAS_PA = False
    print("  [WARN] pyarrow/pandas 不可用，跳过内容核实")

# 优先核实这些：隐性仇恨（同构性最强）+ 上下文讽刺（难度来源）
PRIORITY = [
    "SetFit/implicit_hate_v1",
    "michellemwange25/hate_speech_offensive",
    "tasksource/figlang2020-sarcasm",
    "siddharthyadev/sarcasm-2.0-reddit",
    "macabdul9/SarcasmDetection_Mustard",
    "yl2342/friends_chandler_bing_sarcasm",
]

detail = []
if HAS_PA:
    for ds_id in PRIORITY:
        rec = next((x for x in found if x["id"] == ds_id), None)
        if rec is None:
            print(f"\n  [跳过] {ds_id} 不可用")
            continue
        print(f"\n{'─'*106}")
        print(f"【{ds_id}】 ({rec['kind']})")
        base_url = f"{BASE}/{rec['kind']}/{ds_id}/resolve/main"
        # 找数据文件
        data_files = [f for f in rec["files"]
                      if str(f).endswith((".parquet", ".jsonl", ".json", ".csv", ".tsv"))]
        if not data_files:
            print(f"  未找到数据文件，清单: {rec['files'][:6]}")
            continue
        for rel in data_files[:4]:
            url = f"{base_url}/{rel}"
            try:
                req = urllib.request.Request(url, method="HEAD",
                                             headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    cl = r.headers.get("Content-Length")
                size_mb = int(cl) / 1024 / 1024 if cl else None
            except Exception as e:
                print(f"  [FAIL HEAD] {rel}: {type(e).__name__} {str(e)[:60]}")
                continue
            print(f"  {rel:<62} {size_mb:>8.2f} MB" if size_mb else
                  f"  {rel:<62} 大小未知")
            if size_mb is None or size_mb > 30:
                # 大文件只读 footer
                if str(rel).endswith(".parquet") and size_mb:
                    try:
                        req = urllib.request.Request(
                            url, headers={"User-Agent": "Mozilla/5.0",
                                          "Range": "bytes=-8"})
                        with urllib.request.urlopen(req, timeout=60) as r:
                            tail8 = r.read()
                        if tail8[-4:] == b"PAR1":
                            flen = int.from_bytes(tail8[:4], "little")
                            need = 8 + flen
                            req2 = urllib.request.Request(
                                url, headers={"User-Agent": "Mozilla/5.0",
                                              "Range": f"bytes=-{need}"})
                            with urllib.request.urlopen(req2, timeout=120) as r2:
                                chunk = r2.read()
                            meta = pq.read_metadata(io.BytesIO(b"PAR1" + chunk))
                            print(f"      行数 {meta.num_rows:,}  "
                                  f"列 {meta.schema.names}")
                            detail.append({"id": ds_id, "file": rel,
                                           "size_mb": round(size_mb, 2),
                                           "rows": meta.num_rows,
                                           "columns": meta.schema.names})
                    except Exception as e:
                        print(f"      [footer 解析失败] {type(e).__name__}: {str(e)[:60]}")
                continue
            # 小文件完整下载
            try:
                data = get(url) if False else None
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    raw = r.read()
            except Exception as e:
                print(f"      [下载失败] {type(e).__name__}: {str(e)[:60]}")
                continue
            try:
                if str(rel).endswith(".parquet"):
                    df = pq.read_table(io.BytesIO(raw)).to_pandas()
                elif str(rel).endswith((".jsonl", ".json")):
                    df = pd.read_json(io.BytesIO(raw), lines=str(rel).endswith(".jsonl"))
                elif str(rel).endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(raw))
                else:
                    df = pd.read_csv(io.BytesIO(raw), sep="\t")
            except Exception as e:
                print(f"      [解析失败] {type(e).__name__}: {str(e)[:60]}")
                continue
            print(f"      行数 {len(df):,}   列 {list(df.columns)}")
            # 标签分布
            for col in df.columns:
                cl_ = str(col).lower()
                if cl_ in ("label", "labels", "class", "y", "sarcastic",
                           "is_sarcastic", "subtask_a", "subtask_b", "target"):
                    vc = df[col].value_counts(dropna=False)
                    print(f"      标签列 [{col}] 共 {df[col].nunique()} 类:")
                    for k, v in list(vc.items())[:8]:
                        print(f"          {str(k)[:34]:<36} {v:>7,} "
                              f"({v/len(df)*100:>5.1f}%)")
                    break
            # 是否含上下文（contextual sarcasm 的难度来源）
            ctx_cols = [c for c in df.columns
                        if str(c).lower() in ("context", "conversation", "parent",
                                              "history", "thread", "response")]
            print(f"      上下文字段: {ctx_cols if ctx_cols else '无（单句任务）'}")
            # 文本样例与长度
            text_col = next((c for c in df.columns
                             if str(c).lower() in ("text", "sentence", "headline",
                                                   "content", "tweet", "comment")), None)
            if text_col:
                wl = df[text_col].astype(str).str.split().str.len()
                print(f"      文本列 [{text_col}] 词数: 均值 {wl.mean():.1f} "
                      f"中位 {wl.median():.0f} 最大 {wl.max()}")
                print(f"      样例:")
                for t in df[text_col].astype(str).head(3):
                    print(f"          • {t[:120]}")
            detail.append({"id": ds_id, "file": rel, "size_mb": round(size_mb, 2),
                           "rows": len(df), "columns": list(df.columns),
                           "context_cols": ctx_cols})

with open("experiments/results/hard_task_detail.json", "w", encoding="utf-8") as f:
    json.dump(detail, f, ensure_ascii=False, indent=2, default=str)
print("\n" + "=" * 106)
print("结果已写入: hard_task_availability.json / hard_task_detail.json")
print("=" * 106)
