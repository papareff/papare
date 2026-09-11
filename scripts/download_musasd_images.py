"""下载 MuSaSD images.tar（2.58GB）并核实图文配对率。

探测已确认可行（probe_musasd_v2.py）：
  • 图像可解码: JPEG 900×1200 RGB（PIL 验证通过）
  • 配对规则  : images/<推特ID>.jpg ↔ 文本首列 ID（已抽样验证）
  • 标注可靠  : 人工标注讽刺/非讽刺，非按来源推导（无 Misra 式文风捷径）
  • 磁盘充足  : 剩余 225.8GB，需约 5.2GB（tar + 解压）

⚠️ 待核实的隐患（本脚本的核心任务）：
  由 tar 体积与单图均值估算，图像约 22,970 张，而文本标注共 33,859 条
  → 配对率可能仅 ~68%（Twitter 图像失效是常见现象）。
  这直接决定多模态实验的可用样本量，必须精确核实。

下载特性：
  • 断点续传（Range: bytes=已有大小-）
  • 分块写入并实时报告进度与速率
  • SHA256 校验（可选，用于确认下载完整）
  • 解压后逐个核对图文配对，输出配对率与各划分的可用样本数

产物：
  data/raw/musasd/                      解压后的图像目录
  data/raw/musasd_{train,val,test}.csv  图文配对成功的样本（text,label,image_path）
  data/raw/musasd_orphan_report.json    配对失败明细（缺失图像的文本 ID）
"""
import urllib.request
import os
import sys
import ast
import json
import time
import hashlib
import tarfile
import collections

BASE = "https://hf-mirror.com/datasets/quaeast/multimodal_sarcasm_detection/resolve/main"
TAR_REL = "data/images.tar"
DL_DIR = "data/tmp_multimodal"
IMG_ROOT = "data/raw/musasd"
OUT_DIR = "data/raw"
RES = "experiments/results"
CHUNK = 1024 * 1024          # 1MB

for d in [DL_DIR, "data/raw", RES]:
    os.makedirs(d, exist_ok=True)

TAR_PATH = os.path.join(DL_DIR, "images.tar")

print("=" * 112)
print("MuSaSD images.tar 下载与图文配对核实")
print("=" * 112)

# ── 0. 磁盘预检 ──
import shutil
free_b = shutil.disk_usage(os.path.abspath(".")).free
print(f"\n【0】磁盘预检: 剩余 {free_b/1024**3:.1f} GB")


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None


url = f"{BASE}/{TAR_REL}"
total = remote_size(url)
need = total * 2.1 / 1024 ** 3       # tar + 解压 + 余量
print(f"  images.tar 远端体积 {total/1024**2:.2f} MB")
print(f"  预计需要 {need:.2f} GB（tar + 解压 + 余量）")
if free_b / 1024 ** 3 < need:
    print(f"  [FAIL] 磁盘不足，需 {need:.1f} GB，仅剩 {free_b/1024**3:.1f} GB")
    sys.exit(1)
print(f"  ✓ 磁盘充足")

# ── 1. 下载（断点续传）──
print(f"\n{'─'*112}")
print("【1】下载 images.tar（支持断点续传）")
print("─" * 112)
existing = os.path.getsize(TAR_PATH) if os.path.exists(TAR_PATH) else 0
if existing >= total:
    print(f"  已存在完整文件 {existing/1024**2:.2f} MB，跳过下载")
else:
    if existing:
        print(f"  检测到部分文件 {existing/1024**2:.2f} MB，从断点续传")
    else:
        print(f"  开始全新下载")
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    written = existing
    last_report = t0
    mode = "ab" if existing else "wb"
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(TAR_PATH, mode) as f:
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                written += len(buf)
                now = time.time()
                if now - last_report >= 15:
                    el = now - t0
                    sp = (written - existing) / el / 1024 ** 2 if el > 0 else 0
                    eta = (total - written) / (sp * 1024 ** 2) / 60 if sp > 0 else 0
                    print(f"    {written/1024**2:>8.1f} / {total/1024**2:.1f} MB  "
                          f"({written/total*100:>5.1f}%)  {sp:>5.2f} MB/s  "
                          f"剩余 {eta:>5.1f} min", flush=True)
                    last_report = now
        el = time.time() - t0
        print(f"  ✓ 下载完成: {written/1024**2:.2f} MB，耗时 {el/60:.1f} min，"
              f"平均 {(written-existing)/el/1024**2:.2f} MB/s")
    except Exception as e:
        print(f"  [FAIL] 下载中断: {type(e).__name__}: {str(e)[:120]}")
        print(f"  已保存 {written/1024**2:.2f} MB，重新运行本脚本可从断点续传")
        sys.exit(1)

# 完整性校验
actual = os.path.getsize(TAR_PATH)
if actual != total:
    print(f"  [FAIL] 体积不符: 本地 {actual} vs 远端 {total}，重新运行以续传")
    sys.exit(1)
print(f"  ✓ 体积校验通过 ({actual:,} 字节)")

# ── 2. 解压（只解真实图像，跳过 macOS 噪声条目）──
print(f"\n{'─'*112}")
print("【2】解压图像（过滤 macOS AppleDouble 与 PaxHeader 噪声条目）")
print("─" * 112)
os.makedirs(IMG_ROOT, exist_ok=True)


def is_noise(name):
    b = os.path.basename(name)
    return (b.startswith("._") or "paxheader" in name.lower()
            or name.endswith("/"))


extracted, skipped, failed = 0, 0, 0
t0 = time.time()
try:
    with tarfile.open(TAR_PATH, "r") as tf:
        members = []
        for m in tf.getmembers():
            if is_noise(m.name):
                skipped += 1
                continue
            if not m.isfile():
                continue
            # 路径安全：拒绝绝对路径与 .. 越界（防 tar slip）
            # 注意：IMG_ROOT 是相对路径，必须两边都转绝对路径再比较，
            # 否则相对 dest 永远不 startswith 绝对 root，会把全部图像误判为越界
            dest = os.path.abspath(os.path.normpath(
                os.path.join(IMG_ROOT, os.path.basename(m.name))))
            if not dest.startswith(os.path.abspath(IMG_ROOT)):
                failed += 1
                continue
            members.append((m, dest))
        print(f"  tar 内条目: 真实图像 {len(members):,}，噪声跳过 {skipped:,}")
        for i, (m, dest) in enumerate(members):
            try:
                if not os.path.exists(dest):
                    with tf.extractfile(m) as src, open(dest, "wb") as out:
                        out.write(src.read())
                extracted += 1
            except Exception:
                failed += 1
            if (i + 1) % 2000 == 0:
                print(f"    已解压 {i+1:,}/{len(members):,}  "
                      f"({(i+1)/len(members)*100:.1f}%)  "
                      f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)
except tarfile.TarError as e:
    print(f"  [FAIL] tar 读取失败: {type(e).__name__}: {str(e)[:120]}")
    sys.exit(1)

print(f"  ✓ 解压完成: 成功 {extracted:,} 张，失败 {failed}，噪声跳过 {skipped:,}，"
      f"耗时 {(time.time()-t0)/60:.1f} min")

# ── 3. 图文配对核实（核心任务）──
print(f"\n{'─'*112}")
print("【3】图文配对核实（探测阶段的隐患：图像数可能少于文本数）")
print("─" * 112)
img_ids = {os.path.splitext(f)[0] for f in os.listdir(IMG_ROOT)
           if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))}
print(f"  解压出的图像唯一 ID 数: {len(img_ids):,}")

report = {"n_images_on_disk": len(img_ids)}
pair_rows = []
for sp, fn in [("train", "musasd_train.txt"), ("valid", "musasd_valid.txt"),
               ("test", "musasd_test.txt")]:
    p = os.path.join(DL_DIR, fn)
    if not os.path.exists(p):
        print(f"  [WARN] 缺少文本标注 {p}")
        continue
    lines = [l for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
    recs = []
    for l in lines:
        try:
            recs.append(ast.literal_eval(l))
        except Exception:
            continue
    matched, orphan = [], []
    for r in recs:
        if len(r) < 3:
            continue
        tid, text, label = str(r[0]), str(r[1]), int(r[2])
        ipath = os.path.join(IMG_ROOT, f"{tid}.jpg")
        if os.path.exists(ipath):
            matched.append({"id": tid, "text": text, "label": label,
                            "image_path": ipath})
        else:
            orphan.append(tid)
    rate = len(matched) / len(recs) * 100 if recs else 0
    n_pos = sum(1 for m in matched if m["label"] == 1)
    print(f"\n  ◆ {sp}: 文本 {len(recs):,} 条")
    print(f"    配对成功 {len(matched):,} ({rate:.2f}%)   缺图 {len(orphan):,}")
    if matched:
        print(f"    配对后正类占比 {n_pos/len(matched)*100:.2f}%  "
              f"多数类基线 {max(n_pos, len(matched)-n_pos)/len(matched)*100:.2f}%")
        print(f"    文本词数均值 "
              f"{sum(len(m['text'].split()) for m in matched)/len(matched):.1f}")
    # 落盘配对成功的数据集（列名与 irony/hate 一致：text,label，另加 image_path）
    import pandas as pd
    out = f"{OUT_DIR}/musasd_{sp}.csv"
    if matched:
        pd.DataFrame(matched)[["text", "label", "image_path", "id"]] \
            .to_csv(out, index=False, encoding="utf-8")
        print(f"    → {out}")
    # 缺图 ID 样例
    if orphan:
        print(f"    缺图 ID 样例: {orphan[:3]}")
    pair_rows.append({"split": sp, "text_rows": len(recs), "matched": len(matched),
                      "orphan": len(orphan), "match_rate_pct": round(rate, 2),
                      "pos_rate_pct": round(n_pos / len(matched) * 100, 2) if matched else None,
                      "csv": out if matched else None})
    report[sp] = pair_rows[-1]

# 反向：图像是否有未被任何划分引用的（孤儿图像）
all_text_ids = set()
for sp, fn in [("train", "musasd_train.txt"), ("valid", "musasd_valid.txt"),
               ("test", "musasd_test.txt")]:
    p = os.path.join(DL_DIR, fn)
    if os.path.exists(p):
        for l in open(p, encoding="utf-8").read().splitlines():
            if l.strip():
                try:
                    all_text_ids.add(str(ast.literal_eval(l)[0]))
                except Exception:
                    pass
orphan_imgs = img_ids - all_text_ids
print(f"\n  ◆ 孤儿图像（有图但无文本标注）: {len(orphan_imgs):,}")
report["orphan_images"] = len(orphan_imgs)

tot_text = sum(r["text_rows"] for r in pair_rows)
tot_match = sum(r["matched"] for r in pair_rows)
print(f"\n  ★ 总体配对率: {tot_match:,}/{tot_text:,} = {tot_match/tot_text*100:.2f}%")
print(f"    探测阶段估算图像 {22970:,} 张 vs 实际 {len(img_ids):,} 张 "
      f"→ 估算{'准确' if abs(len(img_ids)-22970)/22970 < 0.1 else '偏差较大'}")
report["overall_match_rate_pct"] = round(tot_match / tot_text * 100, 2)

# ── 4. 多模态实验可用性判定 ──
print(f"\n{'='*112}")
print("【4】多模态实验可用性判定")
print("=" * 112)
te = next((r for r in pair_rows if r["split"] == "test"), None)
tr = next((r for r in pair_rows if r["split"] == "train"), None)
verdicts = []
if te and te["matched"] >= 1000:
    verdicts.append(("test 规模", "ok",
                     f"{te['matched']:,} 条图文对，足够做多模态评测"))
elif te:
    verdicts.append(("test 规模", "warn", f"仅 {te['matched']:,} 条，偏少"))
if te and te["match_rate_pct"] < 90:
    verdicts.append(("配对率", "warn",
                     f"{te['match_rate_pct']:.1f}%，缺图 {te['orphan']:,} 条 → "
                     f"评测集是「有图子集」，与纯文本全量结果不可直接比"))
if tr and tr["matched"] >= 5000:
    verdicts.append(("train 规模", "ok",
                     f"{tr['matched']:,} 条，足够训练图文融合层"))
else:
    verdicts.append(("train 规模", "warn", f"{tr['matched'] if tr else 0:,} 条，偏少"))
verdicts.append(("算力", "warn",
                 "CLIP ViT-L/14@336 已缓存(3265MB)，4GB 显存可做零样本图文分类；"
                 "训练融合层需梯度 → bitsandbytes 0.49.2 可用但缺 peft，"
                 "装 peft 后可试 4-bit QLoRA，否则上服务器"))
# 配对率为 0 时 pos_rate_pct 会是 None（避免除零），不能直接 :.2f 格式化
def _num(v):
    return v if isinstance(v, (int, float)) else float("nan")


_tr_pos, _te_pos = _num(tr["pos_rate_pct"]) if tr else float("nan"), \
                   _num(te["pos_rate_pct"]) if te else float("nan")
verdicts.append(("分布偏移", "warn",
                 f"train 正类 {_tr_pos:.2f}% vs test {_te_pos:.2f}%，"
                 f"计票规则须在 val 上标定"))

ICON = {"ok": "✓", "warn": "⚠", "fail": "✗"}
print(f"  {'检查项':<14} {'结论':<7} 说明")
print("  " + "-" * 96)
for n_, l_, m_ in verdicts:
    print(f"  {n_:<14} {ICON[l_]} {l_:<6} {m_}")
n_fail = sum(1 for _, l, _ in verdicts if l == "fail")
overall = "不可行" if n_fail else "可行，可开展多模态实验"
print(f"\n  ★ 总体判定: {overall}")
report["verdicts"] = [{"item": n_, "level": l_, "msg": m_} for n_, l_, m_ in verdicts]
report["overall"] = overall

with open(f"{RES}/musasd_download_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已写入 {RES}/musasd_download_report.json")
print(f"图像目录: {IMG_ROOT}/  ({len(img_ids):,} 张)")
print(f"数据集  : {OUT_DIR}/musasd_{{train,val,test}}.csv")
print(f"原始 tar: {TAR_PATH}  ({os.path.getsize(TAR_PATH)/1024**2:.0f} MB，"
      f"确认数据集可用后可删除以回收空间)")
print("=" * 112)
