import torch.nn as nn
from transformers import DistilBertModel


class TextEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 7)
    
    def forward(self, ids, mask):
        x = self.distilbert(ids, mask).last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
