import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        attn_scores = self.attn_weights(x)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        weighted_sum = torch.sum(x * attn_weights, dim=1)  # (batch_size, hidden_size)
        return weighted_sum

class FeatureAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.attn_weights = nn.Linear(input_size, input_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        attn_scores = self.attn_weights(x)
        attn_weights = F.softmax(attn_scores, dim=2)
        return x * attn_weights

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout=0.3, attn_type='dual'):
        """
        A hybrid model combining CNN, BiLSTM, and dual attention for RUL prediction.

        Args:
            input_dim (int): Number of input features per timestep.
            hidden_size (int): Hidden size for the BiLSTM layer.
            dropout (float): Dropout rate for regularization.
            attn_type (str): Type of attention - 'none', 'temporal', 'feature', 'dual'.
        """
        super().__init__()
        self.attn_type = attn_type

        # Multi-scale CNN feature extractor
        self.conv3 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)

        self.layernorm = nn.LayerNorm(192)

        # BiLSTM layer
        self.rnn = nn.LSTM(
            input_size=192,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Attention mechanisms
        if attn_type in ['temporal', 'dual']:
            self.temp_attn = TemporalAttention(hidden_size * 2)
        if attn_type in ['feature', 'dual']:
            self.feat_attn = FeatureAttention(input_dim)

        self.dropout = nn.Dropout(dropout)

        # Regression head
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, aux_input=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_dim).
            aux_input (Tensor or None): Optional auxiliary input (e.g., mode_id), shape (batch, aux_dim).

        Returns:
            Tensor: Predicted RUL values, shape (batch,)
        """
        if self.attn_type in ['feature', 'dual']:
            x = self.feat_attn(x)

        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)

        # Multi-scale CNN feature extraction
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)
        x = torch.cat([x1, x2, x3], dim=1)  # (batch, 192, seq_len)
        x = F.relu(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 192)

        x = self.layernorm(x)
        x = self.dropout(x)  # Pre-RNN dropout

        # BiLSTM layer
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_size*2)

        # Attention
        if self.attn_type == 'temporal':
            out = self.temp_attn(rnn_out)
        elif self.attn_type == 'dual':
            out = self.temp_attn(rnn_out)
        else:
            out = rnn_out[:, -1, :]  # Last timestep

        # Final regression layers
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out.squeeze(-1)  # (batch,)
