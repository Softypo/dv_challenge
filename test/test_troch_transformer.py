import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, dim_feedforward=512):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 768*768*1280)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = x.view(-1, 768, 768, 1280)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# Create a random volume for demonstration purposes
volume = torch.randn(1, 1, 768, 768, 1280)

# Create a Transformer model and pass the volume through it
transformer = Transformer()
output = transformer(volume)

# Print the shape of the output
print(output.shape)
