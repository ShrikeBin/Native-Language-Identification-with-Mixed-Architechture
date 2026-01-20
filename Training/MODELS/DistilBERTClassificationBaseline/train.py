import torch
import pandas as pd
from datasets import Dataset as HFDataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from evaluate import load

# ===== CONFIG =====
MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "text"
LABEL_COL = "language"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load full data =====
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = Path(__file__).resolve().parent.parent / "train_config.json"

def load_df(path: str) -> pd.DataFrame:
    path = PROJECT_ROOT / path 
    print(f"Loading data from {path}...")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported data format: {path.suffix}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

train_df = load_df(cfg["train_path"])
test_df = load_df(cfg["test_path"])

label_map = {int(k): v for k, v in cfg["label_map"].items()}
label_map = {v: int(k) for k, v in label_map.items()}
train_df[LABEL_COL] = train_df[LABEL_COL].map(label_map).astype(int)
test_df[LABEL_COL] = test_df[LABEL_COL].map(label_map).astype(int)

# ===== Tokenizer =====
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    texts = [str(t) for t in batch[TEXT_COL]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

# ===== HuggingFace Dataset =====
train_dataset = HFDataset.from_pandas(train_df).shuffle(seed=42)
test_dataset = HFDataset.from_pandas(test_df)
# ==== Tokenize ====
train_dataset = train_dataset.map(tokenize, batched=True, num_proc=8)
test_dataset = test_dataset.map(tokenize, batched=True, num_proc=8)
# ==== Rename label column ====
train_dataset = train_dataset.rename_column(LABEL_COL, "labels")
test_dataset = test_dataset.rename_column(LABEL_COL, "labels")


# ===== Model =====
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_map)
)

# ===== Helper Function =====
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"\nTrainable parameters: {trainable_params:,} / {all_params:,} "
        f"({100 * trainable_params / all_params:.8f}%)")

print_trainable_parameters(model)

# ===== Train Only Classifier Head =====
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

print_trainable_parameters(model)

model.to(DEVICE)

# ===== Metrics =====
accuracy = load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# ===== Data collator =====
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== Training args =====
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-3, # try
    per_device_train_batch_size=96,
    per_device_eval_batch_size=96,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(), # important for memory
    gradient_accumulation_steps=2,
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ===== Train & Evaluate =====
trainer.train()
results = trainer.evaluate()
print("Evaluation results:", results)

# ===== Save model =====
trainer.save_model("./model")
