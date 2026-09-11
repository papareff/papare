"""核实 FIGLANG / hate / musasd 三个泛化任务的数据完整性。"""
import os
import pandas as pd

FILES = [
    "figlang_twitter_train", "figlang_twitter_val", "figlang_twitter_test",
    "figlang_twitter_train_raw", "figlang_twitter_test_raw",
    "figlang_twitter_train_responseonly", "figlang_twitter_test_responseonly",
    "figlang_reddit_train", "figlang_reddit_val", "figlang_reddit_test",
    "figlang_reddit_train_raw", "figlang_reddit_test_raw",
    "figlang_reddit_train_responseonly", "figlang_reddit_test_responseonly",
    "hate_train", "hate_val", "hate_test",
    "musasd_train", "musasd_val", "musasd_test",
]

print("=" * 108)
print("泛化任务数据完整性核实")
print("=" * 108)
missing = []
for f in FILES:
    p = f"data/raw/{f}.csv"
    if not os.path.exists(p):
        missing.append(f)
        print(f"  {f:<38} [缺失]")
        continue
    d = pd.read_csv(p)
    cols = list(d.columns)
    # _raw 版本是三列原始格式（context/response/label），没有 text 列
    if "text" not in cols:
        pos = d["label"].mean() * 100 if "label" in cols else float("nan")
        print(f"  {f:<38} {len(d):>7,} 正类 {pos:>6.2f}%  "
              f"[原始三列格式，无 text]  cols={cols}")
        continue
    pos = d["label"].mean() * 100
    wc = d["text"].astype(str).str.split().str.len()
    has_img = "image_path" in cols
    print(f"  {f:<38} {len(d):>7,} 正类 {pos:>6.2f}%  "
          f"词数均 {wc.mean():>5.1f} max {wc.max():>4}  "
          f"{'含图像列' if has_img else ''}  cols={cols}")

print("\n" + "-" * 108)
print(f"缺失文件: {missing or '无'}")

# 检查 FIGLANG reddit 是否缺 val
print("\n【关键】FIGLANG 验证集可用性（训练需要 val 做早停与选优）")
for src in ["twitter", "reddit"]:
    v = f"data/raw/figlang_{src}_val.csv"
    t = f"data/raw/figlang_{src}_train.csv"
    if os.path.exists(v):
        print(f"  {src:<8} val 存在: {len(pd.read_csv(v)):,} 条")
    elif os.path.exists(t):
        n = len(pd.read_csv(t))
        print(f"  {src:<8} val 缺失 → 需从 train({n:,}) 切 10% 作 val")
    else:
        print(f"  {src:<8} train 与 val 均缺失")

# 检查截断长度需求：FIGLANG 拼接版含 context，可能需要 256
print("\n【关键】截断长度需求（FIGLANG 拼接版含上下文）")
for f in ["figlang_twitter_train", "figlang_twitter_train_raw",
          "figlang_twitter_train_responseonly", "figlang_reddit_train"]:
    p = f"data/raw/{f}.csv"
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    chars = d["text"].astype(str).str.len()
    # 粗估 token 数 ≈ 字符数 / 4
    est_tok = chars / 4
    print(f"  {f:<38} 字符 max {chars.max():>5.0f}  "
          f"估 token max {est_tok.max():>6.0f}  "
          f"p95 {est_tok.quantile(0.95):>6.0f}  "
          f"→ 建议 max_length {int(max(64, est_tok.quantile(0.99)//32*32+32))}")
