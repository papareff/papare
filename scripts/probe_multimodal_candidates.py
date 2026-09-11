"""探测多模态分类数据集（文本+图像），对应论文模块 IV「跨模态门控融合」。

论文现状（paper/实验记录与结论汇总.md 第 91 行）：
  「模块 IV（跨模态门控融合）完全缺失：图 2.2 为多模态框架（文本+图像、
    跨模态注意力、信息熵正则化、门控融合），当前实现为纯文本分类。
    图与代码严重脱节，必须二选一：补多模态实验（如仇恨模因数据集），
    或修改框架图匹配现有实现。」
  → 用户已明确选择：做多模态分类。故本脚本探测可获取的多模态数据集。

候选任务族（均与当前二分类框架兼容，可复用票数门槛与偏置分析）：
  A. 仇恨模因（hateful memes）—— 论文点名的方向，文本+图像，
     且「图像与文本不一致才构成仇恨」与反讽的「字面≠意图」高度同构
  B. 多模态讽刺（MUSTARD / multimodal sarcasm）—— 视频+文本，前一轮已探到
  C. 图文情感 / 图文蕴含

本地已有可复用的多模态骨干（E:\\huggingface_cache）：
  openai/clip-vit-large-patch14-336   CLIP，图文对齐，4GB 显存可做零样本图文分类
  Salesforce/blip-image-captioning-base  BLIP，图像描述生成
  → 这使「本地跑 CLIP 零样本多模态基线」成为可能，无需服务器
"""
import urllib.request
import json
import os

BASE = "https://hf-mirror.com"

# (数据集 ID, 任务族, 模态, 为什么适合)
CANDIDATES = [
    # ── A. 仇恨模因（论文点名方向）──
    ("Multimodal-Fatima/OK-Hateful_memes", "A 仇恨模因", "文本+图像",
     "OK-Hateful，Facebook 仇恨模因衍生，规模适中"),
    ("Multimodal-Fatima/Hateful_memes", "A 仇恨模因", "文本+图像",
     "Facebook Hateful Memes 镜像"),
    ("daniel2588/hateful_memes", "A 仇恨模因", "文本+图像", "Hateful Memes 另一份"),
    ("MMInstruction/Hateful-Memes", "A 仇恨模因", "文本+图像", "指令格式仇恨模因"),
    ("clip-benchmark/wds_hateful_memes", "A 仇恨模因", "文本+图像",
     "webdataset 格式，适配 CLIP 评测"),
    ("nateraw/hateful-memes", "A 仇恨模因", "文本+图像", "Hateful Memes"),
    ("Multimodal-Fatima/MAMI", "A 厌女模因", "文本+图像",
     "MAMI 厌女模因识别，SemEval 2022 共享任务"),
    # ── B. 多模态讽刺（前一轮已探到，此处核实文本+图像可用性）──
    ("macabdul9/SarcasmDetection_Mustard", "B 多模态讽刺", "视频+文本",
     "MUSTARD，仅 test 200 条且含音频 → 规模不足"),
    ("quaeast/multimodal_sarcasm_detection", "B 多模态讽刺", "文本+图像",
     "MuSaSD / Multimodal sarcasm，文本+图像"),
    ("DynamicSuperb/SarcasmDetection_Mustard", "B 多模态讽刺", "视频+文本",
     "MUSTARD 另一份"),
    # ── C. 图文情感 / 蕴含 ──
    ("clip-benchmark/wds_vtab-eurosat", "C 参照", "图像", "纯图像，不适用"),
    ("nlphuji/flickr30k", "C 图文检索", "文本+图像", "图文检索，非分类"),
]


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def probe(ds_id):
    for kind in ["datasets", "models"]:
        try:
            info = json.loads(get(f"{BASE}/api/{kind}/{ds_id}"))
            sib = info.get("siblings") or []
            files = [str(s.get("rfilename")) for s in sib]
            img_ext = (".jpg", ".jpeg", ".png", ".webp", ".zip", ".tar", ".parquet")
            return {
                "id": ds_id, "exists": True, "kind": kind,
                "downloads": info.get("downloads", 0),
                "likes": info.get("likes", 0),
                "n_files": len(files),
                "files": files[:16],
                "has_image_files": any(f.lower().endswith(img_ext) for f in files),
                "n_parquet": sum(1 for f in files if f.endswith(".parquet")),
                "tags": [t for t in info.get("tags", [])
                         if not t.startswith(("region", "size_categories"))][:14],
            }
        except Exception:
            continue
    return {"id": ds_id, "exists": False}


print("=" * 108)
print("多模态数据集探测（论文模块 IV：跨模态门控融合）")
print("=" * 108)

results = []
for ds_id, family, modal, why in CANDIDATES:
    r = probe(ds_id)
    r.update({"family": family, "modal": modal, "why": why})
    results.append(r)
    if not r["exists"]:
        print(f"\n  [不可用] {ds_id}")
        continue
    print(f"\n  [OK {r['kind']}] {ds_id}")
    print(f"    任务族 {family}   模态 {modal}   下载量 {r['downloads']}   "
          f"点赞 {r['likes']}   文件 {r['n_files']}")
    print(f"    含图像类文件: {'是' if r['has_image_files'] else '否'}   "
          f"parquet {r['n_parquet']} 个")
    print(f"    文件: {', '.join(r['files'][:6])}")
    if r["tags"]:
        print(f"    标签: {', '.join(r['tags'][:9])}")
    print(f"    适配理由: {why}")

avail = [r for r in results if r["exists"]]
print(f"\n\n{'=' * 108}")
print(f"可用候选 {len(avail)}/{len(CANDIDATES)}")
print("=" * 108)

# 优先级排序：A 仇恨模因（论文点名）> B 多模态讽刺 > C
PRIO = {"A 仇恨模因": 1, "A 厌女模因": 2, "B 多模态讽刺": 3, "C 图文检索": 8, "C 参照": 9}
avail.sort(key=lambda r: (PRIO.get(r["family"], 6), -int(r.get("downloads") or 0)))
print(f"  {'优先级':<7} {'数据集 ID':<44} {'任务族':<14} {'含图像':>7} {'下载量':>8}")
print("  " + "-" * 88)
for r in avail:
    pr = PRIO.get(r["family"], 6)
    print(f"  {pr:<7} {r['id']:<44} {r['family']:<14} "
          f"{'是' if r['has_image_files'] else '否':>7} {r['downloads']:>8}"
          f"{'  ★' if pr <= 2 else ''}")

# 本地已有多模态骨干核查
print(f"\n\n{'=' * 108}")
print("本地已有多模态骨干（决定能否本地跑零样本基线）")
print("=" * 108)
hub = r"E:\huggingface_cache\hub"
if os.path.isdir(hub):
    for name in sorted(os.listdir(hub)):
        if not name.startswith("models--"):
            continue
        if any(k in name.lower() for k in ["clip", "blip", "vit", "siglip"]):
            root = os.path.join(hub, name)
            sz = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fs in os.walk(root) for f in fs) / 1024 / 1024
            print(f"  [已缓存] {name.replace('models--','').replace('--','/'):<48} "
                  f"{sz:>8.1f} MB")
else:
    print(f"  [WARN] 缓存目录不存在: {hub}")

print("""
  ◆ CLIP 零样本多模态基线的本地可行性
    openai/clip-vit-large-patch14-336 已缓存，ViT-L/14@336 约 890M 参数。
    FP16 权重约 1.8GB，4GB 显存下做【图文零样本分类】可行（无梯度、无反向传播）。
    → 可本地建立多模态基线：CLIP 零样本 vs CLIP+文本裁决者融合 vs 门控融合
    → 但训练跨模态融合层（需梯度）仍需服务器，或改用 4-bit QLoRA（bitsandbytes 0.49.2 已可用）
""")

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/multimodal_candidates.json", "w",
          encoding="utf-8") as f:
    json.dump({"candidates": results,
               "available": [r["id"] for r in avail],
               "recommended_order": [r["id"] for r in avail if PRIO.get(r["family"], 6) <= 3]},
              f, ensure_ascii=False, indent=2)
print(f"结果已写入 experiments/results/multimodal_candidates.json")
print("注：本脚本只探测存在性与文件清单，未下载数据、未核实真实规模与图像可读性。")
print("=" * 108)
