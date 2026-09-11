"""核查候选数据集真实规模：小文件完整下载，大文件只读 parquet footer。"""
import urllib.request
import io
import os
import json

BASE = "https://hf-mirror.com/datasets"
TMP = "data/tmp_probe"
os.makedirs(TMP, exist_ok=True)

import pyarrow.parquet as pq

TARGETS = [
    ("Lots-of-LoRAs/task386_semeval_2018_task3_irony_detection",
     "data/train-00000-of-00001.parquet"),
    ("Lots-of-LoRAs/task386_semeval_2018_task3_irony_detection",
     "data/test-00000-of-00001.parquet"),
    ("siddharthyadev/sarcasm-2.0-reddit", "data/train-00000-of-00001.parquet"),
    ("Heschmat/news-headlines-dataset-sarcasm-detection",
     "data/train-00000-of-00001.parquet"),
    ("CreativeLang/SARC_Sarcasm",
     "data/train-00000-of-00004-b43fbca2a8c4851e.parquet"),
]


def req(url, headers=None, timeout=60):
    h = {"User-Agent": "Mozilla/5.0"}
    if headers:
        h.update(headers)
    return urllib.request.Request(url, headers=h)


def size_of(url):
    with urllib.request.urlopen(req(url, timeout=30)) as r:
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None


def fetch(url):
    with urllib.request.urlopen(req(url), timeout=300) as r:
        return r.read()


print("=" * 100)
print("候选数据集真实规模核查")
print("=" * 100)

results = []
for ds_id, rel in TARGETS:
    url = f"{BASE}/{ds_id}/resolve/main/{rel}"
    print(f"\n{'─'*100}")
    print(f"【{ds_id}】 {rel}")
    try:
        total = size_of(url)
    except Exception as e:
        print(f"  [FAIL] HEAD: {type(e).__name__}: {e}")
        continue
    if total is None:
        print("  [FAIL] 无法获取大小")
        continue
    mb = total / 1024 / 1024
    print(f"  大小: {mb:.2f} MB")

    try:
        if mb < 60:
            data = fetch(url)
            meta = pq.read_metadata(io.BytesIO(data))
            tbl = pq.read_table(io.BytesIO(data))
            df = tbl.to_pandas()
            print(f"  行数: {meta.num_rows:,}")
            print(f"  列名: {list(df.columns)}")
            # 找标签列并统计分布
            for col in df.columns:
                if col.lower() in ("label", "sarcastic", "labels", "class", "y"):
                    vc = df[col].value_counts().to_dict()
                    print(f"  标签列 [{col}] 分布: {vc}")
                    n = len(df)
                    for k, v in sorted(vc.items(), key=lambda x: -x[1]):
                        print(f"      {k}: {v} ({v/n*100:.1f}%)")
                    break
            # 样例
            print(f"  样例首行: {json.dumps({k: str(v)[:90] for k, v in df.iloc[0].items()}, ensure_ascii=False)[:500]}")
            if mb < 40:
                out = f"{TMP}/{ds_id.replace('/','__')}_{os.path.basename(rel)}.parquet"
                with open(out, "wb") as f:
                    f.write(data)
                print(f"  [已缓存] {out}")
        else:
            # 大文件：只读 footer
            tail8 = fetch(url) if False else None
            r = urllib.request.urlopen(req(url, {"Range": f"bytes={total-8}-{total-1}"}), timeout=60)
            tail8 = r.read()
            if tail8[4:] != b"PAR1":
                print("  [WARN] 非标准 parquet 结尾，跳过")
                continue
            flen = int.from_bytes(tail8[:4], "little")
            need = 8 + flen
            r2 = urllib.request.urlopen(
                req(url, {"Range": f"bytes={total-need}-{total-1}"}), timeout=120)
            chunk = r2.read()
            pseudo = b"PAR1" + chunk
            meta = pq.read_metadata(io.BytesIO(pseudo))
            print(f"  行数: {meta.num_rows:,}")
            print(f"  列名: {meta.schema.names}")
            print(f"  行组数: {meta.num_row_groups}")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")

print("\n" + "=" * 100)
print("完成")
print("=" * 100)
