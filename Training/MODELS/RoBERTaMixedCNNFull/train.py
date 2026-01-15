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
            nn.Conv1d(hidden_size, 512, 1), nn.GELU(), nn.Dropout(self.dropout),
            nn.Conv1d(512, 2048, 3), nn.GELU(), nn.Dropout(self.dropout)
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
        
        self.robert = RobertaForSequenceClassification.from_pretrained("../RoBERTaClassificationFull/model")
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
MODEL_NAME = "roberta-base"
TEXT_COL = "text"
LABEL_COL = "language"
MAX_LEN = 256
BATCH_SIZE = 128
NUM_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load full data =====
train_df = pd.read_csv("../../DATA/NEW/Lang-8/train.csv")
test_df = pd.read_csv("../../DATA/NEW/Lang-8/test.csv")


# ==== Map labels ====
label_map = {
    0: "English",
    1: "German",
    2: "Netherlandic",
    3: "Nordic",
    4: "French",
    5: "Italian",
    6: "Portuguese",
    7: "Spanish",
    8: "Russian/Ukrainian",
    9: "Polish/Czech/Slovak",
    10: "Balkan",
    11: "Turkic",
    12: "Chinese",
    13: "Vietnamese",
    14: "Koreanic",
    15: "Japonic",
    16: "Tai",
    17: "Indonesian",
    18: "Uralic",
    19: "Arabic",
    20: "Indo-Iranian"
}
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
model = MixedClassifier(num_classes=21)
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
