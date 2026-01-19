import torch
import pandas as pd
from datasets import Dataset as HFDataset
from transformers import (
    DebertaV2Tokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from evaluate import load
from peft import LoraConfig, get_peft_model

# ===== CONFIG =====
MODEL_NAME = "microsoft/deberta-v3-large"
TEXT_COL = "text"
LABEL_COL = "language"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

# ===== Load full data =====
train_df = pd.read_parquet("../../DATA/OLD/deletedUnknown/train_balanced.parquet")
test_df = pd.read_parquet("../../DATA/OLD/validate.parquet")

# ==== Map labels ====
label_map = {0: "English", 1: "German", 2: "Nordic", 3: "French", 4: "Italian", 5: "Portuguese", 6: "Spanish", 7: "Russian", 8: "Polish", 9: "Other Slavic", 10: "Turkic", 11: "Chinese", 12: "Vietnamese", 13: "Koreanic", 14: "Japonic", 15: "Tai", 16: "Indonesian", 17: "Uralic", 18: "Arabic", 19: "Indo-Iranian"}
label_map = {v: int(k) for k, v in label_map.items()}
train_df[LABEL_COL] = train_df[LABEL_COL].map(label_map).astype(int)
test_df[LABEL_COL] = test_df[LABEL_COL].map(label_map).astype(int)

# ===== Tokenizer =====
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch[TEXT_COL], padding="max_length", truncation=True, max_length=256)

# ===== HuggingFace Dataset =====
train_dataset = HFDataset.from_pandas(train_df).shuffle(seed=42)
test_dataset = HFDataset.from_pandas(test_df)
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset = train_dataset.rename_column(LABEL_COL, "labels")
test_dataset = test_dataset.rename_column(LABEL_COL, "labels")

# ===== Model =====
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_map)
)

# ===== LoRA Config =====
lora_config = LoraConfig(
    r=24,                      
    lora_alpha=48,
    target_modules=[
        "query_proj",
        "key_proj",
        "value_proj",
    ],
    lora_dropout=0.1,
    modules_to_save=["classifier", "pooler"]
)
model = get_peft_model(model, lora_config)

# Train only classifier head + LoRA
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(DEVICE)
model.print_trainable_parameters()

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
    learning_rate=1e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=4,
    weight_decay=0.01,
    gradient_accumulation_steps=12,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ===== Train & Evaluate =====
trainer.train()
results = trainer.evaluate()
print("Evaluation results:", results)

# ===== Save model =====
trainer.save_model("./model")
