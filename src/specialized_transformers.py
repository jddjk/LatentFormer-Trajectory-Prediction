import torch
import torch.nn as nn
from transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer


class TrajectoryEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len]
        for layer in self.layers:
            x = layer(x)
        return x
    


class VisionTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, d_model))  # assuming max map size is 256x256

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len]
        for layer in self.layers:
            x = layer(x)
        return x
    

class MultiAgentDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return x
    
class LatentFormer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_enc_layers, num_dec_layers, dropout=0.1):
        super().__init__()
        self.traj_encoder = TrajectoryEncoder(d_model, num_heads, d_ff, num_enc_layers, dropout)
        self.map_encoder = VisionTransformer(d_model, num_heads, d_ff, num_enc_layers, dropout)
        self.decoder = MultiAgentDecoder(d_model, num_heads, d_ff, num_dec_layers, dropout)
        self.output_projection = nn.Linear(d_model, 2)  # Predicting (x, y) coordinates

    def forward(self, past_trajectories, map_data):
        encoded_traj = self.traj_encoder(past_trajectories)
        encoded_map = self.map_encoder(map_data)
        combined_context = torch.cat((encoded_traj, encoded_map), dim=1)
        predictions = self.decoder(encoded_traj, combined_context)
        return self.output_projection(predictions)