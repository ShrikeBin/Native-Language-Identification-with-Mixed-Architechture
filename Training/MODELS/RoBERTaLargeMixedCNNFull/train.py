from torch import nn
import torch
import pandas as pd
from datasets import Dataset as HFDataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from evaluate import load

class RobertCNN(nn.Module):
    def __init__(self, robertaSQ, num_classes=1):
        super().__init__()

        self.dropout = 0.2

        hidden_size = robertaSQ.config.hidden_size
        self.embed = nn.Embedding.from_pretrained(robertaSQ.roberta.embeddings.word_embeddings.weight.clone(), freeze=True)
        for param in robertaSQ.parameters():
            param.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, 1024, 1), nn.GELU(), nn.Dropout(self.dropout),
            nn.Conv1d(1024, 2048, 3), nn.GELU(), nn.Dropout(self.dropout)
        )

        self.pooler = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.embed(input_ids).transpose(1, 2)
        x = self.conv(x)
        x = self.pooler(x).squeeze(-1)
        logits = self.classifier(x)

        loss = None
        if labels != None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}
    
    
class MixedClassifier(nn.Module):
    def __init__(self, num_classes = 20):
        super().__init__()
        
        self.robert = RobertaForSequenceClassification.from_pretrained("../RoBERTaLargeClassificationFull/model")
        self.cnn = RobertCNN(self.robert, num_classes=num_classes)
        
        from safetensors.torch import load_file
        self.cnn.load_state_dict(load_file("./CNN/model.safetensors"))
        
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.robert.parameters():
            param.requires_grad = False
            
        self.gelu = nn.GELU()
        self.classifier = nn.Linear(2*num_classes, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        robert_x = self.robert.forward(input_ids, attention_mask, labels=labels)['logits']
        cnn_x = self.cnn.forward(input_ids, attention_mask, labels=labels)['logits']
        
        x = torch.cat([robert_x, cnn_x], dim=1)
        
        x = self.gelu(x)
        
        logits = self.classifier(x)

        loss = None
        if labels != None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}
        

# ===== CONFIG =====
MODEL_NAME = "roberta-large"
TEXT_COL = "text"
LABEL_COL = "language"
MAX_LEN = 256
BATCH_SIZE = 128
NUM_EPOCHS = 3
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    texts = [str(t) for t in batch[TEXT_COL]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN)

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
model = MixedClassifier(num_classes=len(label_map))
model.to(DEVICE)

# ===== Params ====
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
print_trainable_parameters(model)
    
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
    learning_rate=1e-3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs_language",
    logging_steps=5000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=1,
    report_to="none",
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ===== Train & Evaluate =====
trainer.train()
results = trainer.evaluate()
print("Evaluation results:", results)

# ===== Save model =====
trainer.save_model("./model")
