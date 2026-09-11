"""核查候选大规模反讽/讽刺数据集的实际规模与类别比例。

目标：为裁决者重训练找到样本量 ≥10k 的数据集，并顺带获得自然类别比例的测试集。
方式：读取镜像站数据集元数据与 parquet 清单，不下载全量内容。
"""
import urllib.request
import json

BASE = "https://hf-mirror.com"

CANDIDATES = [
    # (数据集 ID, 预期内容, 领域)
    ("raquiba/Sarcasm_News_Headline", "Misra 28K 新闻标题", "新闻标题"),
    ("Heschmat/news-headlines-dataset-sarcasm-detection", "同上另一份", "新闻标题"),
    ("siddharthyadev/sarcasm-2.0-reddit", "Reddit 讽刺 v2", "Reddit"),
    ("Thewillonline/reddit-sarcasm", "Reddit 讽刺", "Reddit"),
    ("CreativeLang/SARC_Sarcasm", "Reddit SARC 大规模", "Reddit"),
    ("shiv213/Automatic-Sarcasm-Detection-Twitter", "Twitter 讽刺", "Twitter"),
    ("Lots-of-LoRAs/task386_semeval_2018_task3_irony_detection", "SemEval-2018 Task3", "Twitter"),
    ("gimmaru/tweet_eval-irony", "TweetEval irony", "Twitter"),
]


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def probe(ds_id):
    out = {"id": ds_id}
    # 1) 数据集元数据
    try:
        info = json.loads(get(f"{BASE}/api/datasets/{ds_id}"))
        out["downloads"] = info.get("downloads", 0)
        out["likes"] = info.get("likes", 0)
        out["tags"] = [t for t in info.get("tags", []) if not t.startswith("region")]
        sib = info.get("siblings") or []
        out["files"] = [s.get("rfilename") for s in sib][:25]
        out["n_files"] = len(sib)
    except Exception as e:
        out["error"] = f"meta: {type(e).__name__}: {e}"
        return out

    # 2) 自动生成的 splits 行数（HF auto-generated 元数据）
    try:
        raw = get(f"{BASE}/api/datasets/{ds_id}/auto-processed")
        out["auto_processed"] = json.loads(raw)
    except Exception:
        out["auto_processed"] = None
    return out


print("=" * 100)
print("候选数据集规模核查")
print("=" * 100)

for ds_id, desc, domain in CANDIDATES:
    r = probe(ds_id)
    print(f"\n{'─'*100}")
    print(f"【{ds_id}】 {desc}  ({domain})")
    print(f"{'─'*100}")
    if r.get("error"):
        print(f"  [FAIL] {r['error']}")
        continue
    print(f"  下载量 {r.get('downloads')}  点赞 {r.get('likes')}  文件数 {r.get('n_files')}")
    files = r.get("files") or []
    if files:
        print(f"  文件清单: {', '.join(files[:12])}")
    ap = r.get("auto_processed")
    if ap:
        print(f"  自动处理 splits: {json.dumps(ap, ensure_ascii=False)[:600]}")
