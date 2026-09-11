"""探测镜像站上真实存在的大规模反讽/讽刺数据集。"""
import urllib.request
import json

BASE = "https://hf-mirror.com"


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


for term in ["sarcasm", "irony"]:
    print("=" * 90)
    print(f"搜索关键词: {term}")
    print("=" * 90)
    url = f"{BASE}/api/datasets?search={term}&limit=50"
    try:
        _, body = get(url)
        ds = json.loads(body)
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        continue
    print(f"命中 {len(ds)} 个数据集\n")
    rows = sorted(ds, key=lambda d: d.get("downloads", 0) or 0, reverse=True)
    print(f"{'数据集 ID':<52} {'下载量':>9} {'点赞':>6} {'私有':>5}")
    print("-" * 78)
    for d in rows[:30]:
        print(f"{str(d.get('id')):<52} {d.get('downloads', 0):>9} "
              f"{d.get('likes', 0):>6} {str(d.get('private')):>5}")
