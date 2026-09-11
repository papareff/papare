"""在反讽检测数据（tweet_eval irony）上微调 DistilBERT 作为裁决者与微调基线。
标签：0 = not ironic（非反讽），1 = ironic（反讽）。
"""
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
for f in ["data/raw/irony_train.csv", "data/raw/irony_val.csv"]:
    if not os.path.exists(f):
        print(f"❌ 找不到 {f}，请先运行 download_irony.py")
        exit()

print("📂 加载数据...")
train_df = pd.read_csv("data/raw/irony_train.csv")
val_df = pd.read_csv("data/raw/irony_val.csv")

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128  # 推文较短，128 足够
    )


print("🔄 分词中...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 加载模型（二分类：0=not ironic, 1=ironic）
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "not ironic", 1: "ironic"},
    label2id={"not ironic": 0, "ironic": 1},
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary", pos_label=1),
    }


training_args = TrainingArguments(
    output_dir="./models/irony_distilbert",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
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

model.save_pretrained("./models/irony_distilbert")
tokenizer.save_pretrained("./models/irony_distilbert")

print("✅ 微调完成！模型保存在 ./models/irony_distilbert")
print("📊 验证集评估：")
print(trainer.evaluate())
