import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据（注意：AG News 的 CSV 文件没有表头）
train_df = pd.read_csv("data/raw/train.csv", header=None, names=['label', 'title', 'description'])
test_df = pd.read_csv("data/raw/test.csv", header=None, names=['label', 'title', 'description'])

# 合并标题和描述作为文本
train_df['text'] = train_df['title'] + " " + train_df['description']
test_df['text'] = test_df['title'] + " " + test_df['description']

# 标签从 1 开始，转换为从 0 开始
train_df['label'] = train_df['label'] - 1
test_df['label'] = test_df['label'] - 1

# 从训练集划分验证集 (10%)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

# 保存为最终格式，只保留 text 和 label 列
train_df[['text', 'label']].to_csv("data/raw/agnews_train.csv", index=False)
val_df[['text', 'label']].to_csv("data/raw/agnews_val.csv", index=False)
test_df[['text', 'label']].to_csv("data/raw/agnews_test.csv", index=False)

print(f"AG News processed: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")