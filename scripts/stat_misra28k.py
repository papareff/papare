"""统计已缓存的 Misra 28K 新闻标题数据集的标签分布与文本特征。"""
import pandas as pd
import io
import pyarrow.parquet as pq

F = "data/tmp_probe/Heschmat__news-headlines-dataset-sarcasm-detection_train-00000-of-00001.parquet.parquet"

with open(F, "rb") as f:
    df = pq.read_table(io.BytesIO(f.read())).to_pandas()

print("=" * 92)
print("Misra 28K 新闻标题讽刺数据集")
print("=" * 92)
print(f"总行数: {len(df):,}")
print(f"列名: {list(df.columns)}")
print(f"空值: {df.isnull().sum().to_dict()}")

df["is_sarcastic"] = pd.to_numeric(df["is_sarcastic"], errors="coerce")
df = df.dropna(subset=["is_sarcastic"])
df["is_sarcastic"] = df["is_sarcastic"].astype(int)

print(f"\n有效样本: {len(df):,}")
vc = df["is_sarcastic"].value_counts().sort_index()
for k, v in vc.items():
    print(f"  is_sarcastic={k}: {v:,} ({v/len(df)*100:.1f}%)")

# 文本长度分布
df["len"] = df["headline"].astype(str).str.len()
df["words"] = df["headline"].astype(str).str.split().str.len()
print(f"\n文本长度(字符): 均值 {df['len'].mean():.1f}  中位 {df['len'].median():.0f}  "
      f"最大 {df['len'].max()}")
print(f"文本长度(词):   均值 {df['words'].mean():.1f}  中位 {df['words'].median():.0f}  "
      f"最大 {df['words'].max()}")

# 重复检测
dup = df["headline"].duplicated().sum()
print(f"\n完全重复标题: {dup:,} ({dup/len(df)*100:.2f}%)")
print(f"去重后样本: {len(df.drop_duplicates(subset=['headline'])):,}")

print(f"\n各类别样例:")
for lab in [1, 0]:
    print(f"\n--- is_sarcastic={lab} ---")
    for t in df[df.is_sarcastic == lab]["headline"].head(6):
        print(f"  • {t}")

# 与现有 SemEval 推特数据的域对比
print("\n" + "=" * 92)
print("域对比：Misra 新闻标题 vs 当前 SemEval-2018 推特")
print("=" * 92)
cur = pd.read_csv("data/raw/irony_train.csv")
cur["words"] = cur["text"].astype(str).str.split().str.len()
print(f"{'':<22} {'样本数':>10} {'平均词数':>10} {'反讽/讽刺占比':>14}")
print("-" * 60)
print(f"{'SemEval-2018 推特':<20} {len(cur):>10,} {cur['words'].mean():>10.1f} "
      f"{(cur.label==1).mean()*100:>13.1f}%")
print(f"{'Misra 新闻标题':<20} {len(df):>10,} {df['words'].mean():>10.1f} "
      f"{(df.is_sarcastic==1).mean()*100:>13.1f}%")
print(f"\n参数量/样本数比改善: {66_000_000/len(cur):.0f}:1 → {66_000_000/len(df):.0f}:1")
print("（注：新闻标题是单句无上下文，推特反讽依赖 #irony 标签；两者标注机制不同，")
print("      直接混训存在域偏移风险，需先验证跨域可迁移性）")
