import os
import pandas as pd
import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# 检查数据文件
if not os.path.exists("data/raw/agnews_train.csv"):
    print("❌ 找不到训练数据，请先运行 process_agnews.py")
    exit()

print("📂 加载数据...")
train_df = pd.read_csv("data/raw/agnews_train.csv")
val_df = pd.read_csv("data/raw/agnews_val.csv")

# 转换为 Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

# 加载分词器
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

print("🔄 分词中...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 加载模型（4分类）
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
)

# 评估函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# 训练参数
training_args = TrainingArguments(
    output_dir="./models/agnews_distilbert",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("🏋️ 开始训练...")
trainer.train()

# 保存模型
model.save_pretrained("./models/agnews_distilbert")
tokenizer.save_pretrained("./models/agnews_distilbert")

print("✅ 微调完成！模型保存在 ./models/agnews_distilbert")
print("📊 评估结果：")
print(trainer.evaluate())