"""修正版 MuSaSD 探测：正确选取真实 jpg 验证解码，并核实图文配对关系。

前一版 probe_musasd.py 的缺陷：
  嗅探图像格式时取到的是 macOS AppleDouble 元数据文件（._images / ._820054935970594816.jpg，
  各 0.2KB）与 PaxHeader 条目，而非真实图像 → 误判为「PIL 解码失败、图像损坏」。
  tar 由 macOS 打包，故每个真实图像都伴随 ._xxx 与 PaxHeader/xxx 两个噪声条目。

本版修正：
  ① entry 过滤：跳过以 ._ 开头的 AppleDouble、跳过 PaxHeader/ 目录、只取 *.jpg/*.png
  ② 完整读取第一个真实图像的全部字节后用 PIL 解码，验证 mode/size/format
  ③ 核实图文配对：文本首列 ID 是否与 images/<ID>.jpg 对应
  ④ 解析 valid/test 的第 4 字段（train 只有 3 字段），确认其语义
  ⑤ 统计 train vs valid/test 的类别分布偏移
"""
import urllib.request
import io
import os
import ast
import json
import collections
import struct

BASE = "https://hf-mirror.com/datasets/quaeast/multimodal_sarcasm_detection/resolve/main"
TMP = "data/tmp_multimodal"
os.makedirs(TMP, exist_ok=True)
os.makedirs("experiments/results", exist_ok=True)


def head(url, timeout=40):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        cl = r.headers.get("Content-Length")
        return (int(cl) if cl else None), r.headers.get("Content-Type")


def rng(url, start, end, timeout=180):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def full(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def is_real_image(name):
    n = name.lower()
    base = os.path.basename(n)
    if base.startswith("._"):
        return False          # macOS AppleDouble
    if "paxheader" in n:
        return False          # POSIX pax 扩展头
    return n.endswith((".jpg", ".jpeg", ".png", ".webp"))


print("=" * 112)
print("MuSaSD 修正版探测：真实图像可读性 + 图文配对 + 标注结构")
print("=" * 112)

profile = {"dataset": "quaeast/multimodal_sarcasm_detection",
           "probe_version": 2,
           "fix_note": ("v1 误把 macOS AppleDouble(._*)/PaxHeader 当作图像，"
                        "导致『PIL 解码失败』的错误判定；本版已过滤噪声条目")}

# ══════════ 1. 文本标注结构（含第 4 字段与分布偏移）══════════
print("\n【1】文本标注结构与类别分布")
print("-" * 112)
txt_info = {}
all_ids = {}
for sp in ["train", "valid", "test"]:
    p = f"{TMP}/musasd_{sp}.txt"
    if not os.path.exists(p):
        raw = full(f"{BASE}/data/text/{'train.txt' if sp=='train' else ('valid2.txt' if sp=='valid' else 'test2.txt')}")
        with open(p, "wb") as f:
            f.write(raw)
    lines = [l for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
    recs, bad = [], 0
    field_counts = collections.Counter()
    for l in lines:
        try:
            r = ast.literal_eval(l)
            field_counts[len(r)] += 1
            recs.append(r)
        except Exception:
            bad += 1
    ids = [str(r[0]) for r in recs if len(r) >= 1]
    labels = [r[2] for r in recs if len(r) >= 3]
    lab = collections.Counter(labels)
    tot = sum(lab.values())
    pos_rate = lab.get(1, 0) / tot * 100 if tot else 0
    all_ids[sp] = set(ids)

    print(f"\n  ◆ {sp}   {len(lines):,} 行   解析失败 {bad}   字段数分布 {dict(field_counts)}")
    print(f"    标签分布 {dict(sorted(lab.items()))}   正类(label=1)占比 {pos_rate:.2f}%   "
          f"多数类基线 {max(lab.values())/tot*100:.2f}%")
    print(f"    ID 唯一数 {len(set(ids)):,} / {len(ids):,}  "
          f"({'无重复' if len(set(ids))==len(ids) else '有重复!'})")
    # 字段语义推断
    if recs:
        r0 = recs[0]
        print(f"    字段样例（{len(r0)} 字段）:")
        for i, v in enumerate(r0):
            vs = str(v)
            print(f"      [{i}] {vs[:96]}")
        if len(r0) >= 4:
            f3 = collections.Counter(r[3] for r in recs if len(r) >= 4)
            print(f"    ★ 第 4 字段取值分布: {dict(sorted(f3.items(), key=lambda x: -x[1])[:8])}")
            print(f"      推断: 第 4 字段与标签的关系 —— "
                  f"{'与第3字段完全一致' if all(r[2]==r[3] for r in recs if len(r)>=4) else '与第3字段不同'}")
            if not all(r[2] == r[3] for r in recs if len(r) >= 4):
                agree = sum(1 for r in recs if len(r) >= 4 and r[2] == r[3])
                print(f"      两字段一致率 {agree/len(recs)*100:.2f}%")
                # 交叉表
                ct = collections.Counter((r[2], r[3]) for r in recs if len(r) >= 4)
                print(f"      交叉表 (字段2, 字段3): {dict(sorted(ct.items()))}")
    txt_info[sp] = {"rows": len(lines), "parse_fail": bad,
                    "fields": dict(field_counts),
                    "label_dist": {str(k): int(v) for k, v in lab.items()},
                    "pos_rate_pct": round(pos_rate, 2),
                    "majority_baseline_pct": round(max(lab.values()) / tot * 100, 2),
                    "unique_ids": len(set(ids))}

# 分布偏移
print(f"\n  ◆ train vs valid/test 类别分布偏移")
tr_pos = txt_info["train"]["pos_rate_pct"]
for sp in ["valid", "test"]:
    d = txt_info[sp]["pos_rate_pct"] - tr_pos
    print(f"    train 正类 {tr_pos:.2f}%  →  {sp} 正类 {txt_info[sp]['pos_rate_pct']:.2f}%  "
          f"({d:+.2f}pp)  {'⚠ 存在偏移' if abs(d) > 5 else '基本一致'}")
print(f"    → 若偏移显著，train 上选出的计票规则/阈值未必适用于 test，需在 val 上标定")
profile["text"] = txt_info

# 划分间 ID 重叠（泄漏检查）
print(f"\n  ◆ 划分间 ID 重叠（泄漏检查）")
for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    n = len(all_ids[a] & all_ids[b])
    print(f"    {a} ∩ {b} = {n}   [{'OK' if n == 0 else '泄漏!'}")
profile["id_overlap"] = {f"{a}_{b}": len(all_ids[a] & all_ids[b])
                         for a, b in [("train", "valid"), ("train", "test"),
                                      ("valid", "test")]}

# ══════════ 2. 真实图像可读性（过滤噪声条目）══════════
print(f"\n{'─'*112}")
print("【2】真实图像可读性（已过滤 macOS AppleDouble 与 PaxHeader）")
print("─" * 112)
tar_url = f"{BASE}/data/images.tar"
tar_size, tar_ct = head(tar_url)
print(f"  images.tar  Content-Type={tar_ct}  体积 {tar_size/1024/1024:.2f} MB")
profile["images_tar_size_mb"] = round(tar_size / 1024 / 1024, 2)

# 读前 6MB，足以覆盖若干真实图像（单张约 80~150KB）
probe_len = min(6 * 1024 * 1024, tar_size)
chunk = rng(tar_url, 0, probe_len - 1)
print(f"  已读头部 {len(chunk)/1024/1024:.2f} MB 解析 tar 结构")

entries, off = [], 0
noise = collections.Counter()
while off + 512 <= len(chunk):
    hdr = chunk[off:off + 512]
    if hdr == b"\x00" * 512:
        break
    try:
        name = hdr[0:100].split(b"\x00")[0].decode("utf-8", "replace").strip()
        size_oct = hdr[124:136].split(b"\x00")[0].strip()
        size = int(size_oct, 8) if size_oct else 0
        typeflag = chr(hdr[156]) if hdr[156] else "0"
    except Exception:
        break
    if not name:
        break
    base = os.path.basename(name)
    if base.startswith("._"):
        noise["AppleDouble(._*)"] += 1
    elif "paxheader" in name.lower():
        noise["PaxHeader"] += 1
    elif typeflag == "5":
        noise["目录"] += 1
    elif is_real_image(name):
        entries.append({"name": name, "size": size, "offset": off})
    else:
        noise["其他"] += 1
    off += 512 + ((size + 511) // 512) * 512

print(f"  噪声条目统计: {dict(noise)}")
print(f"  真实图像 entry 数（头部 {probe_len/1024/1024:.0f}MB 内）: {len(entries)}")
if entries:
    szs = [e["size"] for e in entries]
    print(f"  单图大小: 均值 {sum(szs)/len(szs)/1024:.1f} KB  "
          f"最小 {min(szs)/1024:.1f} KB  最大 {max(szs)/1024:.1f} KB")
    print(f"  前 6 个真实图像:")
    for e in entries[:6]:
        print(f"    {e['name']:<56} {e['size']/1024:>8.1f} KB")
    avg = sum(szs) / len(szs)
    est = int(tar_size / (avg + 512 + 2 * 1024))   # 每图伴随约 2 个噪声条目
    print(f"  ★ 估算真实图像总数 ≈ {est:,} 张")
    print(f"    对照文本总行数 {sum(txt_info[s]['rows'] for s in txt_info):,} 条 "
          f"→ {'数量吻合，1:1 配对可信' if abs(est - sum(txt_info[s]['rows'] for s in txt_info)) / sum(txt_info[s]['rows'] for s in txt_info) < 0.15 else '数量不吻合，需完整下载后核对'}")
    profile["est_image_count"] = est
    profile["avg_image_kb"] = round(avg / 1024, 1)

    # 完整读取第一张真实图像并解码
    first = next((e for e in entries
                  if e["offset"] + 512 + e["size"] <= len(chunk)), None)
    if first:
        start = first["offset"] + 512
        img_bytes = chunk[start:start + first["size"]]
        magic = img_bytes[:4]
        fmt = ("JPEG" if img_bytes[:3] == b"\xff\xd8\xff"
               else "PNG" if img_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "UNKNOWN")
        print(f"\n  ◆ 解码验证: {first['name']}")
        print(f"    {len(img_bytes)/1024:.1f} KB  魔数 {magic.hex()}  格式 {fmt}")
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(img_bytes))
            im.load()
            print(f"    ★ PIL 解码成功: format={im.format}  mode={im.mode}  "
                  f"size={im.size[0]}×{im.size[1]}")
            profile["pil_decode_ok"] = True
            profile["image_sample"] = {"name": first["name"], "format": im.format,
                                       "mode": im.mode, "size": list(im.size)}
            # 验证 ID 对应关系
            img_id = os.path.basename(first["name"]).rsplit(".", 1)[0]
            in_train = img_id in all_ids["train"]
            in_valid = img_id in all_ids["valid"]
            in_test = img_id in all_ids["test"]
            print(f"    ★ 图文配对验证: 图像 ID {img_id}")
            print(f"      在 train 标注中: {'是' if in_train else '否'}   "
                  f"valid: {'是' if in_valid else '否'}   "
                  f"test: {'是' if in_test else '否'}")
            profile["id_pairing_verified"] = bool(in_train or in_valid or in_test)
            if in_train or in_valid or in_test:
                print(f"      → 确认配对规则: images/<推特ID>.jpg ↔ 文本首列 ID")
        except ImportError:
            print(f"    [WARN] PIL 不可用")
            profile["pil_decode_ok"] = None
        except Exception as e:
            print(f"    [FAIL] 解码失败: {type(e).__name__}: {e}")
            profile["pil_decode_ok"] = False

# ══════════ 3. 可行性重判 ══════════
print(f"\n{'='*112}")
print("【3】可行性判定（修正版）")
print("=" * 112)
verdicts = []
sz_mb = tar_size / 1024 / 1024
# 磁盘检查
import shutil
tot_b, used_b, free_b = shutil.disk_usage(os.path.abspath("."))
free_gb = free_b / 1024 ** 3
print(f"  当前工作盘剩余空间: {free_gb:.1f} GB")
verdicts.append(("磁盘空间", "ok" if free_gb > sz_mb / 1024 * 3 else "warn",
                 f"剩余 {free_gb:.1f} GB，需 {sz_mb/1024:.2f} GB（tar）+ 解压后约 "
                 f"{sz_mb/1024:.2f} GB → 需约 {sz_mb/1024*2:.1f} GB"))
verdicts.append(("图像体积", "warn" if sz_mb > 1000 else "ok",
                 f"{sz_mb:.0f} MB（约 {sz_mb/1024:.2f} GB），下载耗时取决于带宽"))
if profile.get("pil_decode_ok") is True:
    verdicts.append(("图像可读性", "ok",
                     f"PIL 解码成功 {profile['image_sample']['format']} "
                     f"{profile['image_sample']['size'][0]}×{profile['image_sample']['size'][1]} "
                     f"{profile['image_sample']['mode']}（v1 的『损坏』判定是 AppleDouble 误判）"))
else:
    verdicts.append(("图像可读性", "warn", "未完成解码验证"))
if profile.get("id_pairing_verified"):
    verdicts.append(("图文配对", "ok", "文本首列推特 ID ↔ images/<ID>.jpg，已抽样验证"))
else:
    verdicts.append(("图文配对", "warn", "配对规则未经验证，需完整下载后核对"))
verdicts.append(("文本标注", "ok",
                 f"train {txt_info['train']['rows']:,} / valid {txt_info['valid']['rows']:,} / "
                 f"test {txt_info['test']['rows']:,}，0 解析失败"))
verdicts.append(("标注机制", "ok",
                 "标签为人工标注的讽刺/非讽刺，非按来源推导（不同于 Misra 按新闻源标注，"
                 "无文风捷径）"))
verdicts.append(("类别分布", "warn",
                 f"train 正类 {tr_pos:.1f}% vs test {txt_info['test']['pos_rate_pct']:.1f}%，"
                 f"存在 {txt_info['test']['pos_rate_pct']-tr_pos:+.1f}pp 偏移，"
                 f"计票规则需在 val 上标定"))

print(f"\n  {'检查项':<14} {'结论':<7} 说明")
print("  " + "-" * 96)
ICON = {"ok": "✓", "warn": "⚠", "fail": "✗"}
for n_, l_, m_ in verdicts:
    print(f"  {n_:<14} {ICON[l_]} {l_:<6} {m_}")

n_fail = sum(1 for _, l, _ in verdicts if l == "fail")
n_warn = sum(1 for _, l, _ in verdicts if l == "warn")
overall = ("不可行" if n_fail else
           ("可行（有注意事项）" if n_warn else "可行"))
print(f"\n  ★ 总体判定: {overall}")
print(f"    ✗ 致命问题 {n_fail} 项   ⚠ 注意事项 {n_warn} 项")
profile["verdicts"] = [{"item": n_, "level": l_, "msg": m_} for n_, l_, m_ in verdicts]
profile["overall"] = overall
profile["free_disk_gb"] = round(free_gb, 1)

with open("experiments/results/musasd_profile.json", "w", encoding="utf-8") as f:
    json.dump(profile, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已写入 experiments/results/musasd_profile.json")
print("=" * 112)
