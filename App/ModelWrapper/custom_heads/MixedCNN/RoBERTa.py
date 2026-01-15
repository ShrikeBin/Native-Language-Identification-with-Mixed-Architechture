import torch
from torch import nn
from transformers import RobertaForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, roberta_path, cnn_path, num_classes = 20):
        super().__init__()
        
        self.robert = RobertaForSequenceClassification.from_pretrained(roberta_path)
        self.cnn = RobertCNN(self.robert, num_classes=num_classes)
        
        from safetensors.torch import load_file
        self.cnn.load_state_dict(load_file(cnn_path))
        
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