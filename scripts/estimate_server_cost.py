"""服务器租赁成本估算与平台对比。

依据：本机 RTX 3050 Laptop (4GB) 上的实测耗时，按算力比换算到各 GPU。
"""
import math

# ── 本机实测基准（RTX 3050 Laptop, 4GB VRAM）──
BASE_GPU = "RTX 3050 Laptop"
# 1.5B 单条推理（含辩论全流程）≈ 23.8s（SST-2 pilot50 平均）
# 7B 单条推理（跨设备卸载）≈ 137.1s（探针实测）
# DistilBERT 微调 2575 条 × 3 epoch ≈ 93s（本机实测）
T_15B_DEBATE = 23.8   # s/sample
T_7B_DEBATE = 137.1    # s/sample
T_FT_IRONY = 93        # s for full fine-tune on irony train

# ── 算力换算系数（相对 3050 Laptop）──
# 来源：公开 benchmark + 实际经验；FP16 推理吞吐比
SPEEDUP = {
    "RTX 3050 Laptop": 1.0,
    "RTX 3090":        4.5,
    "RTX 4090":        7.0,
    "A10 (24GB)":      5.0,
    "A40 (48GB)":      8.0,
    "A100 (80GB)":     14.0,
    "H100 (80GB)":     22.0,
}

# ── 实验清单与本机预估耗时 ──
EXPERIMENTS = [
    {
        "name": "E-A: 门控全量验证（1.5B, n=784）",
        "desc": "反讽测试集全量跑中立化辩论 + 门控扫描",
        "base_seconds": 784 * T_15B_DEBATE,
        "gpu_min_vram": 8,
    },
    {
        "name": "E-B: 门控全量验证（7B, n=784）",
        "desc": "7B 辩论者全量验证偏置是否随规模消除 + 门控",
        "base_seconds": 784 * T_7B_DEBATE,
        "gpu_min_vram": 24,
    },
    {
        "name": "E-C: AG News 裁决者微调",
        "desc": "DistilBERT 在 AG News 训练集上微调（~120k 条 × 3 epoch）",
        "base_seconds": T_FT_IRONY * (120000 / 2575),  # 线性缩放
        "gpu_min_vram": 8,
    },
    {
        "name": "E-D: AG News 门控扫描（1.5B, n=7600 测试集）",
        "desc": "AG News 测试集跑中立化辩论 + 门控扫描",
        "base_seconds": 7600 * T_15B_DEBATE,
        "gpu_min_vram": 8,
    },
    {
        "name": "E-E: RAG 独立增益消融（反讽, n=300）",
        "desc": "隔离检索器贡献：纯 ICL vs ICL+检索 vs 无示例",
        "base_seconds": 300 * 3 * T_15B_DEBATE,  # 3 条件
        "gpu_min_vram": 8,
    },
    {
        "name": "E-F: 模块 IV 跨模态（仇恨模因, 1.5B）",
        "desc": "数据准备 + CLIP 编码 + 多模态辩论 + 评测",
        "base_seconds": 3600 * 6,  # 粗估 6h on 3050
        "gpu_min_vram": 16,
    },
]

# ── 平台价格表（2026-09 参考价，元/小时）──
PLATFORMS = {
    "AutoDL": {
        "RTX 3090":   {"price": 1.88, "vram": 24},
        "RTX 4090":   {"price": 2.58, "vram": 24},
        "A10":        {"price": 3.20, "vram": 24},
        "A40":        {"price": 4.80, "vram": 48},
        "A100":       {"price": 8.80, "vram": 80},
    },
    "恒源云": {
        "RTX 3090":   {"price": 2.00, "vram": 24},
        "RTX 4090":   {"price": 2.80, "vram": 24},
        "A10":        {"price": 3.50, "vram": 24},
        "A100":       {"price": 9.50, "vram": 80},
    },
    "阿里云 PAI-DSW": {
        "A10":        {"price": 5.60, "vram": 24},
        "A100":       {"price": 15.00, "vram": 80},
    },
    "腾讯云 TI": {
        "A10":        {"price": 5.20, "vram": 24},
        "A100":       {"price": 14.50, "vram": 80},
    },
}

def pick_gpu(experiment, platform_prices):
    """为该实验选该平台最便宜的满足显存要求的 GPU"""
    min_vram = experiment["gpu_min_vram"]
    candidates = [(gpu, info) for gpu, info in platform_prices.items()
                  if info["vram"] >= min_vram]
    if not candidates:
        return None, None, None
    best = min(candidates, key=lambda x: x[1]["price"])
    return best[0], best[1]["price"], best[1]["vram"]


lines = []
W = lines.append

W("=" * 90)
W("服务器租赁成本估算报告")
W("基准：RTX 3050 Laptop (4GB) 实测耗时 → 按算力比换算")
W("=" * 90)
W("")

# ── 单项实验在各平台的成本 ──
for exp in EXPERIMENTS:
    W("-" * 90)
    W(f"【{exp['name']}】")
    W(f"  说明: {exp['desc']}")
    W(f"  最低显存需求: {exp['gpu_min_vram']}GB")
    base_h = exp["base_seconds"] / 3600
    W(f"  本机(3050)预估: {base_h:.2f}h")
    W("")
    W(f"  {'平台':<16} {'GPU':<14} {'显存':>5} {'单价':>7} {'换算耗时':>9} {'成本':>8}")
    W(f"  {'-'*16} {'-'*14} {'-'*5} {'-'*7} {'-'*9} {'-'*8}")
    for pname, gpus in PLATFORMS.items():
        gpu, price, vram = pick_gpu(exp, gpus)
        if gpu is None:
            W(f"  {pname:<16} {'— 无满足显存要求的机型 —':^50}")
            continue
        speedup = SPEEDUP.get(gpu, 1.0)
        hours = base_h / speedup
        cost = hours * price
        W(f"  {pname:<16} {gpu:<14} {vram:>4}G ¥{price:>5.2f}/h {hours:>8.2f}h ¥{cost:>7.2f}")
    W("")

# ── 全部实验打包成本 ──
W("=" * 90)
W("【全部实验打包估算】")
W("=" * 90)
total_base_h = sum(e["base_seconds"] for e in EXPERIMENTS) / 3600
W(f"  本机(3050)总预估: {total_base_h:.1f}h ≈ {total_base_h/24:.1f}天")
W("")
W(f"  {'平台':<16} {'推荐GPU':<14} {'显存':>5} {'单价':>7} {'换算耗时':>9} {'总成本':>8}")
W(f"  {'-'*16} {'-'*14} {'-'*5} {'-'*7} {'-'*9} {'-'*8}")
for pname, gpus in PLATFORMS.items():
    # 取能满足所有实验的最高显存需求的 GPU
    max_vram = max(e["gpu_min_vram"] for e in EXPERIMENTS)
    gpu, price, vram = pick_gpu({"gpu_min_vram": max_vram}, gpus)
    if gpu is None:
        # 退而求其次：取该平台最高配
        gpu = max(gpus.keys(), key=lambda g: gpus[g]["vram"])
        price = gpus[gpu]["price"]
        vram = gpus[gpu]["vram"]
    speedup = SPEEDUP.get(gpu, 1.0)
    hours = total_base_h / speedup
    cost = hours * price
    W(f"  {pname:<16} {gpu:<14} {vram:>4}G ¥{price:>5.2f}/h {hours:>8.2f}h ¥{cost:>7.2f}")

W("")
W("=" * 90)
W("【平台对比总结】")
W("=" * 90)
W("  AutoDL:")
W("    ✓ 价格最低，GPU 型号最全，开箱即用 PyTorch/CUDA 镜像")
W("    ✓ 支持 JupyterLab / SSH / VSCode Remote，学术用户首选")
W("    ✓ 按秒计费，不用可暂停不计费，适合间歇性实验")
W("    ✗ 高峰期热门卡可能排队；不适合需要长期稳定运行的生产任务")
W("")
W("  恒源云:")
W("    ✓ 价格与 AutoDL 接近，部分冷门卡库存更充裕")
W("    ✓ 支持 Docker 自定义镜像")
W("    ✗ 文档与社区不如 AutoDL 活跃")
W("")
W("  阿里云 PAI-DSW / 腾讯云 TI:")
W("    ✓ 企业级稳定性，SLA 保障，适合论文投稿前的最终验证")
W("    ✓ 可与对象存储/数据库无缝集成，适合大规模数据管线")
W("    ✗ 单价约为 AutoDL 的 2~3 倍；开通流程较复杂（需企业认证或个人实名）")
W("")
W("  自有实验室服务器:")
W("    ✓ 零边际成本，长期使用最划算")
W("    ✓ 数据不出内网，合规性最好")
W("    ✗ 需自行维护环境；若排队严重则实际可用时间不确定")
W("")
W("=" * 90)
W("【建议方案】")
W("=" * 90)
W("  第一阶段（验证门控显著性 + RAG 消融）:")
W("    → AutoDL RTX 4090，预计 E-A + E-E 合计约 6~8h，成本 ¥15~21")
W("")
W("  第二阶段（7B 全量 + AG News 泛化）:")
W("    → AutoDL A40 或 RTX 4090，预计 E-B + E-C + E-D 合计约 12~18h，成本 ¥30~47")
W("")
W("  第三阶段（跨模态，可选）:")
W("    → AutoDL A40 (48GB)，预计 E-F 约 1~2h，成本 ¥5~10")
W("")
W("  总计预算参考: ¥50~80 即可完成全部核心实验")
W("  （以上为 2026-09 参考价，实际以平台实时报价为准）")

txt = "\n".join(lines)
print(txt)

out_path = r"E:\zuhui\project\paper\服务器租赁成本估算.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(txt)
print(f"\n已保存: {out_path}")
