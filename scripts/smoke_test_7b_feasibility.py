"""7B 辩论者可行性冒烟测试：验证 Qwen2-7B-Instruct 能否在 4GB 显存 + 32GB 内存下跑起来。

方案：FP16 不量化 + device_map=auto 自动跨设备放置（显存放不下的层卸载到内存）。
注：INT4 量化 + CPU 卸载在当前库版本存在兼容性缺陷（量化层无法复制到 CPU），故不采用。

只做加载与少量样本测速，不改动任何实验结果。
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch

MODEL = "Qwen/Qwen2-7B-Instruct"

print("=== 加载 Qwen2-7B-Instruct（FP16 + device_map=auto 跨设备放置）===")
from transformers import AutoTokenizer, AutoModelForCausalLM

# 4GB 显存放不下 14.5GB 的 FP16 模型：显存留 3GiB，其余放内存（32GB 足够）
max_memory = {0: "3GiB", "cpu": "25GiB"}

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
load_time = time.time() - t0
print(f"加载耗时: {load_time:.1f}s")

if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU 显存占用: {alloc:.2f} GB")

# 统计各设备上的层数分布（判断卸载程度）
dist = {}
for name, p in model.named_parameters():
    dist[str(p.device)] = dist.get(str(p.device), 0) + 1
print(f"参数设备分布: {dist}")

# 测速：3 条反讽样本的分类式生成
samples = [
    "I just love when my flight gets delayed for the 5th time today.",
    "What a beautiful day to be stuck in traffic for two hours.",
    "The new restaurant downtown has amazing food and great service.",
]
prompt_tpl = (
    "Decide whether the following tweet is ironic or not ironic. "
    "Answer with exactly one line: 'FINAL ANSWER: pro' (ironic) or 'FINAL ANSWER: con' (not ironic).\n"
    "Tweet: {text}\nFINAL ANSWER:"
)

# 跨设备放置时 model.device 不可靠，取词嵌入层所在设备
input_device = model.get_input_embeddings().weight.device

print("\n=== 单条推理测速 ===")
latencies = []
for i, text in enumerate(samples):
    prompt = prompt_tpl.format(text=text)
    inputs = tok(prompt, return_tensors="pt").to(input_device)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.time() - t1
    latencies.append(dt)
    gen = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f"  [{i}] 耗时 {dt:.1f}s  生成: {gen[:60]}")

print(f"\n平均单条耗时: {sum(latencies)/len(latencies):.1f}s")
est = sum(latencies)/len(latencies)
print(f"估算：150 条带辩论（约 4 次生成/条）≈ {150*4*est/3600:.1f} 小时")
