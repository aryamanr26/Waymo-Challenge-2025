# planner/embeddings.py
import torch
import torch.nn as nn


class PoseEmbedding(nn.Module):
    def __init__(self, input_dim=6, seq_len=48, d_model=1024):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * seq_len, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, pose_seq):
        flat = self.flatten(pose_seq)
        emb = self.mlp(flat)
        return emb.unsqueeze(1)


class PoseTokenEncoder(nn.Module):
    def __init__(self, input_dim=64, d_model=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, pose_token):
        emb = self.encoder(pose_token)
        return emb.unsqueeze(1)


class RouteTokenEncoder(nn.Module):
    def __init__(self, input_dim=16, d_model=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, route_token):
        emb = self.encoder(route_token)
        return emb.unsqueeze(1)


class RouteEmbedding(nn.Module):
    def __init__(self, num_intents=4, d_model=1024):
        super().__init__()
        self.embedding = nn.Embedding(num_intents, d_model)

    def forward(self, intent_ids):
        emb = self.embedding(intent_ids)
        return emb.unsqueeze(1)
