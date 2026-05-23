import torch
import torch.nn as nn
from transformers import DistilBertModel


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        scores = torch.softmax(
            self.attention_weights(lstm_output),
            dim=1
        )
        return torch.sum(scores * lstm_output, dim=1)


class MultimodalFusionModel(nn.Module):

    def __init__(self):
        super(
            MultimodalFusionModel,
            self
        ).__init__()

        # ==================================================
        # SPEECH ENCODER
        # ==================================================

        self.conv1 = nn.Conv1d(
            in_channels=120,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.batch_norm1 = nn.BatchNorm1d(
            128
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(
            kernel_size=2
        )
        self.dropout = nn.Dropout(
            0.3
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attention = AttentionLayer(
            hidden_dim=256
        )
        self.speech_fc = nn.Linear(
            256,
            128
        )

        # ==================================================
        # TEXT ENCODER
        # ==================================================

        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # Freeze Lower Layers
        for name, parameter in self.distilbert.named_parameters():
            if "transformer.layer.5" in name:
                parameter.requires_grad = True
            elif "transformer.layer.4" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
        
        self.text_fc = nn.Linear(
            768,
            256
        )

        # ==================================================
        # FUSION CLASSIFIER
        # ==================================================

        self.fusion_fc1 = nn.Linear(
            128 + 256,
            256
        )
        self.fusion_fc2 = nn.Linear(
            256,
            128
        )
        self.output_layer = nn.Linear(
            128,
            7
        )
    
    def forward(
        self,
        speech,
        input_ids,
        attention_mask
    ):

        # ==================================================
        # SPEECH FORWARD PASS
        # ==================================================

        speech = speech.permute(
            0,
            2,
            1
        )
        speech = self.conv1(speech)
        speech = self.batch_norm1(speech)
        speech = self.relu(speech)
        speech = self.maxpool(speech)
        speech = self.dropout(speech)
        speech = speech.permute(
            0,
            2,
            1
        )

        lstm_output, _ = self.bilstm(
            speech
        )
        attention_output = self.attention(
            lstm_output
        )
        speech_embedding = self.speech_fc(
            attention_output
        )
        speech_embedding = self.relu(
            speech_embedding
        )

        # ==================================================
        # TEXT FORWARD PASS
        # ==================================================

        text_outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = text_outputs.last_hidden_state[
            :,
            0,
            :
        ]
        text_embedding = self.text_fc(
            cls_embedding
        )
        text_embedding = self.relu(
            text_embedding
        )

        # ==================================================
        # MULTIMODAL FUSION
        # ==================================================

        fusion_embedding = torch.cat(

            [
                speech_embedding,
                text_embedding
            ],

            dim=1
        )

        fusion_embedding = self.fusion_fc1(
            fusion_embedding
        )

        fusion_embedding = self.relu(
            fusion_embedding
        )

        fusion_embedding = self.dropout(
            fusion_embedding
        )

        fusion_embedding = self.fusion_fc2(
            fusion_embedding
        )

        fusion_embedding = self.relu(
            fusion_embedding
        )

        logits = self.output_layer(
            fusion_embedding
        )

        return logits, fusion_embedding
