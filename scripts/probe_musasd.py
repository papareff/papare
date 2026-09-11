"""核实 quaeast/multimodal_sarcasm_detection (MuSaSD) 的规模与图像可读性。

为什么必须先核实再下载：
  images.tar 体积未知（可能几百 MB），若图像损坏/格式异常/规模不足，
  下载与解压都是浪费。先用 HTTP Range 只读文件头部，确认：
    ① tar 内文件数与总体积（只读 tar 头部若干条 entry）
    ② 图像格式与尺寸（读前几张的头部字节判断魔数，必要时完整读一张解码验证）
    ③ 文本标注文件的规模、标签体系、与图像的对应关系
  论文模块 IV 需要图文配对数据，且标注机制必须可靠
  （Misra 按新闻源标注导致模型靠文风作弊，跨域迁移效率仅 4.5%，是反面教材）。
"""
import urllib.request
import io
import os
import json
import tarfile
import struct

BASE = "https://hf-mirror.com/datasets/quaeast/multimodal_sarcasm_detection/resolve/main"
TMP = "data/tmp_multimodal"
os.makedirs(TMP, exist_ok=True)
os.makedirs("experiments/results", exist_ok=True)

TEXT_FILES = [
    ("train", "data/text/train.txt"),
    ("valid", "data/text/valid2.txt"),
    ("test", "data/text/test2.txt"),
]
IMAGE_TAR = "data/images.tar"


def head(url, timeout=40):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        cl = r.headers.get("Content-Length")
        ct = r.headers.get("Content-Type")
        return (int(cl) if cl else None), ct


def rng(url, start, end, timeout=120):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def full(url, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def sniff_image(buf):
    """由魔数判断图像格式，不解码整图。"""
    if buf[:3] == b"\xff\xd8\xff":
        return "JPEG", None
    if buf[:8] == b"\x89PNG\r\n\x1a\n":
        # PNG IHDR: width/height 在偏移 16/20
        try:
            w, h = struct.unpack(">II", buf[16:24])
            return "PNG", (w, h)
        except Exception:
            return "PNG", None
    if buf[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF", None
    if buf[:4] == b"RIFF" and buf[8:12] == b"WEBP":
        return "WEBP", None
    if buf[:2] == b"BM":
        return "BMP", None
    return "UNKNOWN", None


print("=" * 110)
print("MuSaSD (quaeast/multimodal_sarcasm_detection) 规模与图像可读性核实")
print("=" * 110)

profile = {"dataset": "quaeast/multimodal_sarcasm_detection"}

# ── 1. 文件清单与体积 ──
print("\n【1】文件清单与体积")
print("-" * 110)
api = f"https://hf-mirror.com/api/datasets/quaeast/multimodal_sarcasm_detection"
try:
    info = json.loads(full(api, timeout=40).decode("utf-8", "replace"))
    files = [str(s.get("rfilename")) for s in (info.get("siblings") or [])]
    print(f"  仓库文件数 {len(files)}")
    for f in files:
        print(f"    {f}")
    profile["repo_files"] = files
except Exception as e:
    files = []
    print(f"  [WARN] 无法读取仓库清单: {type(e).__name__}: {e}")

# ── 2. 文本标注文件 ──
print(f"\n{'─'*110}")
print("【2】文本标注文件")
print("─" * 110)
text_profile = {}
for split, rel in TEXT_FILES:
    url = f"{BASE}/{rel}"
    try:
        sz, ct = head(url)
    except Exception as e:
        print(f"  [FAIL HEAD] {split}: {type(e).__name__}: {str(e)[:70]}")
        continue
    print(f"\n  ◆ {split}  ({rel})  {sz/1024:.1f} KB" if sz else
          f"\n  ◆ {split}  ({rel})  大小未知")
    try:
        raw = full(url, timeout=180)
    except Exception as e:
        print(f"    [FAIL 下载] {type(e).__name__}: {str(e)[:70]}")
        continue
    txt = raw.decode("utf-8", "replace")
    lines = [l for l in txt.splitlines() if l.strip()]
    print(f"    行数 {len(lines):,}")
    if lines:
        print(f"    首 3 行原文:")
        for l in lines[:3]:
            print(f"      | {l[:150]}")
        # 推断分隔符
        for sep_name, sep in [("制表符", "\t"), ("竖线", "|"), ("三竖线", "|||"),
                              ("逗号", ","), ("空格", " ")]:
            cnts = [l.count(sep) for l in lines[:50]]
            if cnts and min(cnts) == max(cnts) and min(cnts) > 0:
                print(f"    分隔符推断: {sep_name}（前 50 行每行恰 {min(cnts)} 个）")
                break
        # 标签分布：尝试常见格式
        import collections
        lab_counter = collections.Counter()
        parsed = 0
        for l in lines:
            parts = [p.strip() for p in l.replace("|||", "\t").split("\t")]
            if len(parts) >= 2:
                # 标签通常在首列或末列
                cand = parts[0] if parts[0] in ("0", "1", "sarcasm", "non-sarcasm",
                                                "SARCASM", "NOT_SARCASM") else parts[-1]
                lab_counter[cand] += 1
                parsed += 1
        if parsed:
            print(f"    标签分布（解析 {parsed}/{len(lines)} 行）: "
                  f"{dict(lab_counter.most_common(6))}")
            if lab_counter:
                tot = sum(lab_counter.values())
                mx = max(lab_counter.values())
                print(f"    多数类基线 {mx/tot*100:.2f}%   类别数 {len(lab_counter)}")
        # 图像引用字段
        img_refs = [l for l in lines[:5] if ".jpg" in l.lower() or ".png" in l.lower()]
        print(f"    含图像文件名引用: {'是' if img_refs else '否（前5行未见）'}")
        if img_refs:
            print(f"      样例: {img_refs[0][:150]}")
    text_profile[split] = {"rel": rel, "size_kb": round((sz or 0) / 1024, 1),
                           "lines": len(lines)}
    # 缓存文本文件（小，便于后续直接使用）
    with open(f"{TMP}/musasd_{split}.txt", "wb") as f:
        f.write(raw)
    print(f"    [已缓存] {TMP}/musasd_{split}.txt")

profile["text"] = text_profile

# ── 3. images.tar 头部探测（不下载全量）──
print(f"\n{'─'*110}")
print("【3】images.tar 体积与图像格式（HTTP Range 只读头部，不下载全量）")
print("─" * 110)
tar_url = f"{BASE}/{IMAGE_TAR}"
try:
    tar_size, tar_ct = head(tar_url)
    print(f"  Content-Type: {tar_ct}")
    print(f"  体积: {tar_size/1024/1024:.2f} MB" if tar_size else "  体积: 未知")
    profile["images_tar_size_mb"] = round(tar_size / 1024 / 1024, 2) if tar_size else None
except Exception as e:
    tar_size = None
    print(f"  [FAIL HEAD] {type(e).__name__}: {str(e)[:80]}")
    profile["images_tar_error"] = str(e)[:200]

if tar_size:
    # 读前 2MB，解析 tar entry 头部（每个 header 512 字节）
    probe_len = min(2 * 1024 * 1024, tar_size)
    try:
        chunk = rng(tar_url, 0, probe_len - 1)
        print(f"  已读取头部 {len(chunk)/1024/1024:.2f} MB 用于解析 tar 结构")
        entries = []
        off = 0
        while off + 512 <= len(chunk):
            hdr = chunk[off:off + 512]
            if hdr[:512] == b"\x00" * 512:
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
            entries.append({"name": name, "size": size, "type": typeflag,
                            "offset": off})
            # 跳到下一个 header（512 对齐）
            off += 512 + ((size + 511) // 512) * 512
        print(f"  头部可解析 entry 数: {len(entries)}")
        dirs = [e for e in entries if e["type"] == "5"]
        files_e = [e for e in entries if e["type"] != "5"]
        if dirs:
            print(f"  目录项: {[d['name'] for d in dirs[:6]]}")
        if files_e:
            szs = [e["size"] for e in files_e if e["size"] > 0]
            print(f"  文件项（前 {min(8, len(files_e))} 个）:")
            for e in files_e[:8]:
                print(f"    {e['name']:<52} {e['size']/1024:>8.1f} KB")
            if szs:
                avg = sum(szs) / len(szs)
                est = avg * tar_size / max(sum(e['size'] + 512 + ((e['size']+511)//512)*512 - e['size'] for e in files_e), 1)
                print(f"  平均单文件大小 {avg/1024:.1f} KB")
                print(f"  ★ 由体积粗估图像总数 ≈ {int(tar_size/ (avg + 1024)):,} 张"
                      f"（假设每文件额外 1KB tar 开销）")
                profile["est_image_count"] = int(tar_size / (avg + 1024))
                profile["avg_image_kb"] = round(avg / 1024, 1)
            # 图像格式嗅探：完整读取头部区间内的第一张图
            first = next((e for e in files_e
                          if e["size"] > 0 and e["offset"] + 512 + e["size"] <= len(chunk)),
                         None)
            if first:
                start = first["offset"] + 512
                buf = chunk[start:start + min(first["size"], 64)]
                fmt, dim = sniff_image(buf)
                print(f"  图像格式嗅探（{first['name']}）: {fmt}"
                      + (f"  尺寸 {dim[0]}×{dim[1]}" if dim else ""))
                profile["image_format"] = fmt
                profile["image_dims_sample"] = dim
                # 尝试真正解码验证可读性
                try:
                    from PIL import Image
                    img_bytes = chunk[start:start + first["size"]]
                    im = Image.open(io.BytesIO(img_bytes))
                    im.load()
                    print(f"  ★ PIL 解码成功: mode={im.mode} size={im.size} "
                          f"format={im.format}")
                    profile["pil_decode_ok"] = True
                    profile["pil_mode"] = im.mode
                    profile["pil_size"] = list(im.size)
                except ImportError:
                    print(f"  [WARN] PIL 不可用，无法验证解码（仅魔数嗅探）")
                    profile["pil_decode_ok"] = None
                except Exception as e:
                    print(f"  [FAIL] PIL 解码失败: {type(e).__name__}: {str(e)[:80]}")
                    profile["pil_decode_ok"] = False
        else:
            print(f"  [WARN] 未解析到文件 entry，tar 可能使用非标准头或需完整下载")
    except Exception as e:
        print(f"  [FAIL Range] {type(e).__name__}: {str(e)[:90]}")
        profile["tar_probe_error"] = str(e)[:200]

# ── 4. 可行性判定 ──
print(f"\n{'='*110}")
print("【4】可行性判定")
print("=" * 110)
verdicts = []
sz_mb = (tar_size or 0) / 1024 / 1024
if sz_mb <= 0:
    verdicts.append(("体积未知", "warn", "无法判断，需完整下载后才能确定"))
elif sz_mb < 200:
    verdicts.append(("体积", "ok", f"{sz_mb:.1f} MB，可本地下载"))
elif sz_mb < 1500:
    verdicts.append(("体积", "warn", f"{sz_mb:.1f} MB，较大但可下载（注意磁盘与带宽）"))
else:
    verdicts.append(("体积", "fail", f"{sz_mb:.1f} MB，过大，建议服务器端下载"))
if profile.get("pil_decode_ok") is True:
    verdicts.append(("图像可读性", "ok",
                     f"PIL 解码成功，{profile.get('pil_size')} {profile.get('pil_mode')}"))
elif profile.get("pil_decode_ok") is False:
    verdicts.append(("图像可读性", "fail", "PIL 解码失败，图像可能损坏"))
elif profile.get("image_format") and profile["image_format"] != "UNKNOWN":
    verdicts.append(("图像可读性", "warn",
                     f"魔数为 {profile['image_format']}，但未经解码验证"))
else:
    verdicts.append(("图像可读性", "warn", "未能确认格式"))
if text_profile:
    verdicts.append(("文本标注", "ok", f"{len(text_profile)} 个划分已读取并缓存"))
else:
    verdicts.append(("文本标注", "fail", "未能读取任何文本划分"))

print(f"  {'检查项':<16} {'结论':<8} 说明")
print("  " + "-" * 90)
for name, level, msg in verdicts:
    icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}[level]
    print(f"  {name:<16} {icon} {level:<6} {msg}")

n_fail = sum(1 for _, l, _ in verdicts if l == "fail")
n_warn = sum(1 for _, l, _ in verdicts if l == "warn")
if n_fail:
    overall = "不可行 —— 存在致命问题，不建议下载"
elif n_warn:
    overall = "有条件可行 —— 需先解决警告项（建议完整下载 tar 后再验证图文配对完整性）"
else:
    overall = "可行 —— 建议下载"
print(f"\n  ★ 总体判定: {overall}")
profile["verdicts"] = [{"item": n_, "level": l_, "msg": m_} for n_, l_, m_ in verdicts]
profile["overall"] = overall

with open("experiments/results/musasd_profile.json", "w", encoding="utf-8") as f:
    json.dump(profile, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已写入 experiments/results/musasd_profile.json")
print(f"文本文件已缓存至 {TMP}/（仅文本，未下载 images.tar）")
print("=" * 110)
