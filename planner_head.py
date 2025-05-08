# planner/planner_head.py
import torch
import torch.nn as nn
from transformers import T5EncoderModel


class PlannerHead3D(nn.Module):
    def __init__(self, model_name="google/flan-t5-base", d_model=1024, num_waypoints=20, dropout_rate=0.1, mc_samples=10):
        super().__init__()
        self.t5_encoder = T5EncoderModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(d_model, num_waypoints * 3)
        self.mc_samples = mc_samples

    def forward_once(self, inputs_embeds):
        enc = self.t5_encoder.encoder(inputs_embeds=inputs_embeds)
        cls_repr = enc.last_hidden_state[:, 0, :]
        dropped = self.dropout(cls_repr)
        preds = self.regressor(dropped)
        return preds.view(preds.size(0), -1, 3)

    def forward(self, bev, time_tokens, pose_emb, route_emb):
        inputs = torch.cat([bev, time_tokens, pose_emb, route_emb], dim=1)
        if not self.training and self.mc_samples > 1:
            self.t5_encoder.train()
            samples = [self.forward_once(inputs) for _ in range(self.mc_samples)]
            self.t5_encoder.eval()
            stacked = torch.stack(samples, dim=0)
            return stacked.mean(dim=0), stacked.var(dim=0)
        else:
            pred = self.forward_once(inputs)
            return pred, torch.zeros_like(pred)
