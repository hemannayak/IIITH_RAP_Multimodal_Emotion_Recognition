import torch
import torch.nn as nn


class AttentionLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(
            hidden_dim,
            1
        )
    
    def forward(self, lstm_output):

        # Compute Attention Scores
        attention_scores = self.attention_weights(
            lstm_output
        )

        # Normalize Scores
        attention_scores = torch.softmax(
            attention_scores,
            dim=1
        )

        # Weighted Sum
        context_vector = torch.sum(
            attention_scores * lstm_output,
            dim=1
        )

        return context_vector


class AdvancedSpeechEmotionModel(nn.Module):

    def __init__(self):

        super(
            AdvancedSpeechEmotionModel,
            self
        ).__init__()

        # --------------------------------------------------
        # CNN FEATURE EXTRACTION
        # --------------------------------------------------

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

        # --------------------------------------------------
        # BiLSTM TEMPORAL MODELLING
        # --------------------------------------------------

        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # --------------------------------------------------
        # ATTENTION MECHANISM
        # --------------------------------------------------

        self.attention = AttentionLayer(
            hidden_dim=256
        )

        # --------------------------------------------------
        # DENSE LAYERS
        # --------------------------------------------------

        self.fc1 = nn.Linear(
            256,
            128
        )
        self.fc2 = nn.Linear(
            128,
            7
        )
    
    def forward(self, x):

        # ----------------------------------------------
        # INPUT SHAPE
        # (batch, time_steps, features)
        # ----------------------------------------------

        x = x.permute(
            0,
            2,
            1
        )

        # ----------------------------------------------
        # CNN
        # ----------------------------------------------

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        # ----------------------------------------------
        # PREPARE FOR LSTM
        # ----------------------------------------------

        x = x.permute(
            0,
            2,
            1
        )

        # ----------------------------------------------
        # BiLSTM
        # ----------------------------------------------

        lstm_output, _ = self.bilstm(x)

        # ----------------------------------------------
        # ATTENTION
        # ----------------------------------------------

        attention_output = self.attention(
            lstm_output
        )

        # ----------------------------------------------
        # EMOTION EMBEDDING
        # ----------------------------------------------

        embedding = self.fc1(
            attention_output
        )

        embedding = self.relu(
            embedding
        )

        embedding = self.dropout(
            embedding
        )

        # ----------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------

        logits = self.fc2(
            embedding
        )

        return logits, embedding


# Alias for compatibility
SpeechEmotionModel = AdvancedSpeechEmotionModel
