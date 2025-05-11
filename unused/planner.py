import torch
import torch.nn as nn
from transformers import T5EncoderModel

class PoseEmbedding(nn.Module):
    """
    Pose MLP: encodes past ego status (x, y, vx, vy, ax, ay) sequence into a fixed embedding.
    """
    def __init__(self, input_dim=6, seq_len=48, d_model=1024):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * seq_len, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, pose_seq):
        # pose_seq: (batch_size, seq_len, input_dim)
        flat = self.flatten(pose_seq)
        emb = self.mlp(flat)  # (batch_size, d_model)
        # add sequence dim
        return emb.unsqueeze(1)  # (batch_size, 1, d_model)
    
class PoseTokenEncoder(nn.Module):
    """
    Encodes a pose token of shape (batch, 64) into (batch, 1, 1024)
    """
    def __init__(self, input_dim=64, d_model=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, pose_token):
        emb = self.encoder(pose_token)  # (batch, d_model)
        return emb.unsqueeze(1)         # (batch, 1, d_model)

class RouteEmbedding(nn.Module):
    """
    Routing intent embedding: maps discrete intent IDs to continuous embeddings.
    """
    def __init__(self, num_intents=4, d_model=1024):
        super().__init__()
        self.embedding = nn.Embedding(num_intents, d_model)

    def forward(self, intent_ids):
        # intent_ids: (batch_size,)
        emb = self.embedding(intent_ids)  # (batch_size, d_model)
        return emb.unsqueeze(1)  # (batch_size, 1, d_model)

class PlannerHead3D(nn.Module):
    """
    Planner Head for 3D waypoints (x, y, z) using Flan-T5-Base with MC-Dropout.
    """
    def __init__(
        self,
        model_name="google/flan-t5-base",
        d_model=1024,
        num_waypoints=20,
        dropout_rate=0.1,
        mc_samples=10
    ):
        super().__init__()
        self.t5_encoder = T5EncoderModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        # Regression: predict x, y, z per waypoint
        self.regressor = nn.Linear(d_model, num_waypoints * 3)
        self.mc_samples = mc_samples

    def forward_once(self, inputs_embeds):
        enc = self.t5_encoder.encoder(inputs_embeds=inputs_embeds)
        cls_repr = enc.last_hidden_state[:, 0, :]  # (batch, d_model)
        dropped = self.dropout(cls_repr)
        preds = self.regressor(dropped)
        return preds.view(preds.size(0), -1, 3)  # (batch, num_waypoints, 3)

    def forward(self, bev, time_tokens, pose_emb, route_emb):
        """
        bev: (batch, F, d_model)
        time_tokens: (batch, L_time, d_model)
        pose_emb: (batch, 1, d_model)
        route_emb: (batch, 1, d_model)
        """
        inputs = torch.cat([bev, time_tokens, pose_emb, route_emb], dim=1)
        if not self.training and self.mc_samples > 1:
            # MC-Dropout
            self.t5_encoder.train()
            samples = [self.forward_once(inputs) for _ in range(self.mc_samples)]
            self.t5_encoder.eval()
            stacked = torch.stack(samples, dim=0)
            mean = stacked.mean(dim=0)
            var = stacked.var(dim=0)
            return mean, var
        else:
            pred = self.forward_once(inputs)
            return pred, torch.zeros_like(pred)

class RouteTokenEncoder(nn.Module):
    """
    Encodes a routing token of shape (batch, 16) into (batch, 1, 1024)
    """
    def __init__(self, input_dim=16, d_model=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, route_token):
        emb = self.encoder(route_token)  # (batch, d_model)
        return emb.unsqueeze(1)          # (batch, 1, d_model)


# Example usage:
if __name__ == "__main__":
    batch = 4
    seq_len = 48  # 12 seconds @ 4 Hz
    # pose_seq = torch.randn(batch, seq_len, 6)
    # print(pose_seq.shape) # 4, 32

    pose_token = torch.randn(4, 64)
    pose_token_encoder = PoseTokenEncoder()
    pose_embedding = pose_token_encoder(pose_token)  # (4, 1, 1024)
    print("Pose embedding shape:", pose_embedding.shape)

    route_token = torch.randn(4, 16)

    # Encode token -> embedding
    route_token_encoder = RouteTokenEncoder()
    route_embedding = route_token_encoder(route_token)  # (4, 1, 1024)
    print("Route embedding shape:", route_embedding.shape)
    # intents = torch.randint(0, 4, (batch,))
    # print(intents.shape) # 4, 16
    # pose_module = PoseEmbedding(seq_len=seq_len, d_model=1024)
    # route_module = RouteEmbedding(num_intents=4, d_model=1024)
    # pose_emb = pose_module(pose_seq)       # (batch, 1, d_model)
    # print(pose_emb.shape)
    # route_emb = route_module(intents)      # (batch, 1, d_model)
    # print(route_emb.shape)
    
    # dummy BEV and time tokens
    bev = torch.randn(batch, 256, 1024) #torch.Size([4, 256, 1024])
    time_tokens = torch.randn(batch, 128, 1024) #torch.Size([4, 128, 1024])
    
    planner = PlannerHead3D(model_name="google/flan-t5-large", d_model=1024)
    waypoints_mean, waypoints_var = planner(bev, time_tokens, pose_embedding, route_embedding)
    print("Waypoints mean:", waypoints_mean.shape)  # (batch, 20, 3)
    print("Waypoints variance:", waypoints_var.shape)  # (batch, 20, 3)