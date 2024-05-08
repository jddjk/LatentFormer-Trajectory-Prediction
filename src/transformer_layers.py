import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from positionwise_feedforward import PositionwiseFeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.layer_norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2))
        x2 = self.layer_norm2(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        x2 = self.layer_norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2))
        x2 = self.layer_norm2(x)
        x = x + self.dropout(self.cross_attn(x2, memory, memory))
        x2 = self.layer_norm3(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x
