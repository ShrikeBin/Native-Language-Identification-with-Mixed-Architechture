from torch import nn
import torch
from transformers import AutoModel

# CNN -> Convolutional Neural Network
class CustomCNN(nn.Module):
    def __init__(self, num_classes=1, dropout=0.2):
        super().__init__()

        embedding_model = AutoModel.from_pretrained('roberta-base')
        hidden_size = embedding_model.config.hidden_size
        self.embed = embedding_model.embeddings.word_embeddings

        # If you don't want to use roberta embeddings
        # self.embed = nn.Embedding(50265, 512)

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, 512, 3), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(512, 1024, 3), nn.GELU(), nn.Dropout(dropout)
        )

        self.pooler = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.embed(input_ids).transpose(1, 2)
        x = self.conv(x)
        x = self.pooler(x).squeeze(-1)
        logits = self.classifier(x)

        loss = None
        if labels != None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}
