"""诊断预测坍缩问题：
1. 检查测试集标签分布与数据格式
2. 用真实裁决者模型对干净文本推理，验证模型本身是否正常
3. 模拟拼接辩论文本后的输入，观察预测是否坍缩
4. 检查辩论者实际生成的文本内容
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- 1. 数据检查 ----------
print("=" * 60)
print("1. 测试集数据检查")
print("=" * 60)
df = pd.read_csv("data/splits/sst2_test.csv")
print("columns:", df.columns.tolist())
print("shape:", df.shape)
print("head:")
print(df.head(5))
if 'label' in df.columns:
    print("label value_counts:")
    print(df['label'].value_counts())
text_col = 'sentence' if 'sentence' in df.columns else 'text'
print("text length stats:", df[text_col].str.len().describe())

# ---------- 2. 裁决者对干净文本推理 ----------
print("\n" + "=" * 60)
print("2. 裁决者（DistilBERT SST-2）对干净文本推理")
print("=" * 60)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
print("model loaded, num_labels:", model.config.num_labels)

# 跳过脏行（重复表头）
df['label_num'] = pd.to_numeric(df['label'], errors='coerce')
df_clean = df.dropna(subset=['label_num']).copy()
df_clean['label_num'] = df_clean['label_num'].astype(int)
print(f"去除脏行后样本数: {len(df_clean)}")

samples = df_clean.head(20)
clean_preds = []
for idx, (i, row) in enumerate(samples.iterrows()):
    text = row[text_col]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    clean_preds.append(pred)
    if idx < 5:
        print(f"  [{idx}] true={row['label_num']} pred={pred} logits={logits[0].tolist()}")
acc = sum(1 for (_, r), p in zip(samples.iterrows(), clean_preds) if p == r['label_num']) / len(samples)
print(f"干净输入准确率（前20条）: {acc:.2%}")

# ---------- 3. 模拟拼接辩论文本后的输入 ----------
print("\n" + "=" * 60)
print("3. 拼接辩论文本后的输入（模拟 judge_agent 行为）")
print("=" * 60)
# 模拟辩论历史：2轮，每轮 pro/con 各一段（用占位文本模拟典型长度）
fake_debate = [
    "Pro: This movie is clearly positive because it shows uplifting themes.",
    "Con: I disagree, the plot has dark undertones suggesting negative.",
    "Pro: The acting and direction demonstrate a positive message overall.",
    "Con: Nevertheless the ending is bleak which points to negative."
]
debate_text = " ".join(fake_debate)
if len(debate_text) > 300:
    debate_text = debate_text[:300] + "..."

concat_preds = []
for i, row in samples.iterrows():
    text = row[text_col]
    input_text = f"Original: {text}\nDebate: {debate_text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
    n_tokens = inputs['input_ids'].shape[1]
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    concat_preds.append(pred)
    if i < 5:
        print(f"  [{i}] true={row.get('label','?')} pred={pred} tokens={n_tokens} logits={logits[0].tolist()}")
acc2 = sum(1 for (_, r), p in zip(samples.iterrows(), concat_preds) if p == r['label_num']) / len(samples)
print(f"拼接输入准确率（前20条）: {acc2:.2%}")
print(f"拼接输入预测分布: pro={concat_preds.count(1)}, con={concat_preds.count(0)}")

# 关键诊断：看拼接后原始文本是否被截断丢失
print("\n" + "=" * 60)
print("4. 截断诊断：拼接后原始文本保留情况")
print("=" * 60)
for i, row in samples.iterrows():
    if i >= 3:
        break
    text = row[text_col]
    input_text = f"Original: {text}\nDebate: {debate_text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"  [{i}] 总token数={len(tokens)}, 前20个={tokens[:20]}")
    print(f"       后20个={tokens[-20:]}")

print("\n" + "=" * 60)
print("5. 结论汇总")
print("=" * 60)
print(f"干净输入预测分布: pro={clean_preds.count(1)}, con={clean_preds.count(0)}")
print(f"拼接输入预测分布: pro={concat_preds.count(1)}, con={concat_preds.count(0)}")
