"""本地算力可行性预检：RoBERTa 可达性 + 显存占用实测。

RTX 3050 Laptop 4GB 是硬约束，需实测各配置峰值显存，决定本地/服务器分工。
"""
import os
import sys
import urllib.request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", BASE)

print("=" * 92)
print("本地算力可行性预检")
print("=" * 92)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}")
    print(f"显存总量: {p.total_memory/1024**3:.2f} GB")
    print(f"当前已占用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
else:
    print("[FAIL] 无可用 GPU")

# ── 1. RoBERTa-base 可达性 ──
print("\n" + "─" * 92)
print("【1】roberta-base 模型文件可达性（镜像站）")
print("─" * 92)
files = ["config.json", "vocab.json", "merges.txt", "tokenizer.json",
         "model.safetensors"]
ok = True
for f in files:
    url = f"{BASE}/roberta-base/resolve/main/{f}"
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            sz = f"{int(cl)/1024/1024:.1f} MB" if cl else "未知"
            print(f"  [OK {r.status}] {f:<22} {sz}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] {f:<22} {type(e).__name__}: {e}")
print(f"\n  → roberta-base {'可下载' if ok else '不可下载，需在服务器端获取'}")

# ── 2. 显存实测：前向+反向单步 ──
print("\n" + "─" * 92)
print("【2】显存峰值实测（单步 train，含反向传播与优化器状态）")
print("─" * 92)

CONFIGS = [
    # (骨干, batch_size, max_length, 用途)
    ("distilbert-base-uncased", 32, 64, "阶段1 Misra（原计划）"),
    ("distilbert-base-uncased", 16, 64, "阶段1 降 bs"),
    ("distilbert-base-uncased", 16, 128, "阶段2 SemEval"),
    ("roberta-base", 16, 64, "阶段1 RoBERTa"),
    ("roberta-base", 8, 64, "阶段1 RoBERTa 降 bs"),
    ("roberta-base", 4, 64, "阶段1 RoBERTa 极限"),
]

results = []
for backbone, bs, ml, desc in CONFIGS:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        if not os.path.exists(os.path.expanduser(f"~/.cache")) and backbone == "roberta-base":
            pass
        tok = AutoTokenizer.from_pretrained(backbone)
        model = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=2).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

        texts = ["this is a sample sarcastic headline for memory test"] * bs
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=ml, return_tensors="pt").to("cuda")
        enc["labels"] = torch.randint(0, 2, (bs,)).to("cuda")

        # 预热 + 计时 3 步
        import time
        model.train()
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            out = model(**enc)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        peak = torch.cuda.max_memory_allocated() / 1024**3
        step_t = sum(times) / len(times)
        fits = peak < 3.6  # 留 0.4GB 余量
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        results.append({
            "backbone": backbone, "bs": bs, "max_len": ml, "desc": desc,
            "peak_gb": round(peak, 2), "step_sec": round(step_t, 3),
            "fits_4gb": fits, "params_m": round(n_param, 1),
        })
        print(f"  [{backbone:<24} bs={bs:>2} len={ml:>3}] "
              f"峰值 {peak:>5.2f} GB  单步 {step_t:>5.2f}s  "
              f"{'✓ 本地可行' if fits else '✗ 超出 4GB'}   ({desc})")

        del model, opt, tok, enc
    except RuntimeError as e:
        msg = str(e).split("\n")[0][:80]
        results.append({
            "backbone": backbone, "bs": bs, "max_len": ml, "desc": desc,
            "peak_gb": None, "step_sec": None, "fits_4gb": False,
            "params_m": None, "error": msg,
        })
        print(f"  [{backbone:<24} bs={bs:>2} len={ml:>3}] "
              f"[OOM/ERROR] {msg}   ({desc})")
        torch.cuda.empty_cache()
    except Exception as e:
        results.append({
            "backbone": backbone, "bs": bs, "max_len": ml, "desc": desc,
            "peak_gb": None, "step_sec": None, "fits_4gb": False,
            "params_m": None, "error": f"{type(e).__name__}: {e}"[:100],
        })
        print(f"  [{backbone:<24} bs={bs:>2} len={ml:>3}] "
              f"[FAIL] {type(e).__name__}: {str(e)[:90]}")
        torch.cuda.empty_cache()

# ── 3. 耗时估算 ──
print("\n" + "─" * 92)
print("【3】本地训练耗时估算（基于实测单步时间）")
print("─" * 92)
import math
JOBS = [
    ("阶段1 Misra", 21281, 4),
    ("阶段2 SemEval", 2575, 8),
]
for r in results:
    if not r.get("fits_4gb") or not r.get("step_sec"):
        continue
    print(f"\n  配置: {r['backbone']} bs={r['bs']} len={r['max_len']}")
    total = 0.0
    for jname, n, epochs in JOBS:
        spe = math.ceil(n / r["bs"])
        t = spe * epochs * r["step_sec"] / 60
        total += t
        print(f"    {jname:<16} {spe:>5} 步/epoch × {epochs} = "
              f"{spe*epochs:>5} 步  → {t:>7.1f} min ({t/60:.1f} h)")
    print(f"    {'合计':<16} {'':>5} {'':>10} {'':>5}  → {total:>7.1f} min ({total/60:.2f} h)")

import json
os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/local_feasibility.json", "w", encoding="utf-8") as f:
    json.dump({"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
               if torch.cuda.is_available() else None,
               "roberta_reachable": ok, "configs": results},
              f, ensure_ascii=False, indent=2)
print(f"\n结果已写入 experiments/results/local_feasibility.json")
print("=" * 92)
