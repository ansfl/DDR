import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model) if batch_first=True
        x = x + self.pe.transpose(0, 1)[:, :x.size(1), :]
        return self.dropout(x)

class MagTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super(MagTransformer, self).__init__()
        self.d_model = d_model
        #self.embedding = nn.Linear(input_dim, d_model)
        self.embedding = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding=1)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Conv1d expects (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.embedding(x)
        # Transpose back to (batch, seq_len, d_model) for Transformer
        x = x.transpose(1, 2)
        
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Take the last time step output
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.head(x)
        return x.squeeze(-1)
