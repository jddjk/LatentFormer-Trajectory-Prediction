import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        attention = torch.nn.functional.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.linear(output)
