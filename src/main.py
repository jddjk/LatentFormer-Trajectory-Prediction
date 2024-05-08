import torch
from specialized_transformers import LatentFormer

d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

# Sample data
past_trajectories = torch.randn(32, 10, d_model)  # batch_size=32, sequence_length=10, feature_size=d_model
map_data = torch.randn(32, 256, d_model)  # batch_size=32, map_size=256, feature_size=d_model

# Model
model = LatentFormer(d_model, num_heads, d_ff, num_layers, num_layers)

# Forward pass
predicted_trajectories = model(past_trajectories, map_data)
print(predicted_trajectories.shape)  # Expected output: (32, 10, 2)
