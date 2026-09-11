"""核实 LoRA 参数差异方案所需依赖的真实可用性。

背景：此前用 Glob 检查 site-packages 目录名，报告 "peft/bitsandbytes 均未安装"，
但后续 grep 在 site-packages 中命中了大量 bitsandbytes 文件路径 —— 说明 Glob
的目录名匹配方式不可靠（可能因为 dist-info 命名或大小写）。
必须用 import 实测，因为 LoRA 参数差异方案（正反方辩论者各自 adapter）
是本机 4GB 显存下唯一可行的参数差异路线，其可行性直接决定服务器任务设计。
"""
import importlib
import subprocess
import sys

DEPS = [
    ("peft", "LoRA adapter 核心库，参数差异方案的必需依赖"),
    ("bitsandbytes", "4/8-bit 量化，4GB 显存下做生成式微调的必需依赖"),
    ("torch", "运行时"),
    ("transformers", "模型加载"),
    ("accelerate", "device_map / 量化配置"),
    ("datasets", "数据加载"),
    ("sklearn", "分析（导入名 sklearn，包名 scikit-learn）"),
]

print("=" * 100)
print("LoRA 参数差异方案依赖核实（import 实测，非目录名匹配）")
print("=" * 100)

status = {}
for mod, why in DEPS:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", None)
        if ver is None:
            try:
                from importlib.metadata import version as _v
                pkg = {"sklearn": "scikit-learn"}.get(mod, mod)
                ver = _v(pkg)
            except Exception:
                ver = "unknown"
        status[mod] = {"ok": True, "version": str(ver)}
        print(f"  [可用]   {mod:<16} {ver:<14} {why}")
    except Exception as e:
        status[mod] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        print(f"  [不可用] {mod:<16} {type(e).__name__}: {str(e)[:70]}   {why}")

# torch 能力细查（决定本地能否做 LoRA）
print("\n" + "-" * 100)
print("torch / GPU 能力")
print("-" * 100)
try:
    import torch
    print(f"  torch {torch.__version__}   CUDA 编译版本 {torch.version.cuda}")
    print(f"  cuda 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}   显存 {p.total_memory/1024**3:.2f} GB")
        print(f"  当前占用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB  "
              f"预留: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
        print(f"  bf16 支持: {torch.cuda.is_bf16_supported()}")
        print(f"  计算能力: {p.major}.{p.minor}  "
              f"({'支持 4-bit 量化' if p.major >= 7 else '不支持 4-bit 量化(需 sm_70+)'}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")

# transformers 是否暴露 PeftAdapterMixin（即使 peft 缺失，也说明版本支持）
print("\n" + "-" * 100)
print("transformers 对 PEFT 的内置支持")
print("-" * 100)
try:
    from transformers import PeftAdapterMixin
    print("  [可用] transformers.PeftAdapterMixin —— 支持 load_adapter/set_adapter")
    status["transformers_peft_mixin"] = {"ok": True}
except Exception as e:
    print(f"  [不可用] transformers.PeftAdapterMixin: {type(e).__name__}")
    status["transformers_peft_mixin"] = {"ok": False, "error": str(e)[:100]}

try:
    from transformers.integrations import PeftAdapterMixin as _P2  # noqa
    print("  [可用] transformers.integrations.PeftAdapterMixin")
except Exception:
    pass

# 用 pip 交叉验证（import 失败时确认是否真的未安装）
print("\n" + "-" * 100)
print("pip 元数据交叉验证")
print("-" * 100)
try:
    out = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True, text=True, timeout=120)
    lines = out.stdout.splitlines()
    for key in ["peft", "bitsandbytes", "trl", "torch", "transformers", "accelerate"]:
        hits = [l for l in lines if l.lower().startswith(key + "=")]
        print(f"  {key:<16} {hits[0] if hits else '未安装'}")
        status.setdefault("pip", {})[key] = hits[0] if hits else None
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")

# ── 结论：LoRA 参数差异在本机是否可行 ──
print("\n" + "=" * 100)
print("【结论】LoRA 参数差异方案的本地可行性")
print("=" * 100)
peft_ok = status.get("peft", {}).get("ok", False)
bnb_ok = status.get("bitsandbytes", {}).get("ok", False)
concl = []
if peft_ok and bnb_ok:
    concl.append("peft + bitsandbytes 均可用 → 本机可做 4-bit QLoRA 生成式微调。")
    concl.append("1.5B 模型 4-bit 量化约 1.0~1.2GB，加 LoRA 梯度与优化器状态，"
                 "4GB 显存下 batch=1~2 可能可行，但需实测峰值显存。")
    concl.append("★ 若实测可行，则辩论方参数差异【可在本地完成】，不必上服务器。")
elif peft_ok and not bnb_ok:
    concl.append("peft 可用但 bitsandbytes 不可用 → 只能做 FP16 LoRA（无量化）。")
    concl.append("1.5B FP16 权重约 3.0GB，4GB 显存下做生成式微调【几乎必然 OOM】"
                 "（推理已占 3.5GB，训练还需梯度+优化器状态）。")
    concl.append("→ LoRA adapter 训练必须放服务器；本地只能做 adapter 推理（切换 adapter 仅增几 MB）。")
else:
    concl.append("peft 不可用 → LoRA 方案需先安装 peft。")
    if bnb_ok:
        concl.append("bitsandbytes 可用（此前 Glob 检查的『未安装』结论有误，已用 import 实测纠正）。")
    else:
        concl.append("bitsandbytes 也不可用。")
    concl.append("→ 安装 peft（纯 Python，约 1MB，无 CUDA 编译需求）后，"
                 "若 bitsandbytes 仍缺失则只能 FP16 LoRA，训练仍需上服务器。")

for c in concl:
    print("  " + c)

import json
import os
os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/lora_dependency_check.json", "w",
          encoding="utf-8") as f:
    json.dump({"status": status, "conclusions": concl,
               "correction": ("此前基于 Glob 目录名匹配报告 peft/bitsandbytes 均未安装，"
                              "本次改用 import 实测纠正；Glob 对 site-packages 的目录名"
                              "匹配不可靠（dist-info 命名/大小写差异）")},
              f, ensure_ascii=False, indent=2)
print(f"\n结果已写入 experiments/results/lora_dependency_check.json")
print("=" * 100)
