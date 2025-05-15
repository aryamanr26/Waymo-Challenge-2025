# planner/planner_head.py
import torch
import torch.nn as nn
from transformers import T5EncoderModel


class PlannerHead3D(nn.Module):
    def __init__(self, model_name="google/flan-t5-base", d_model=1024, num_waypoints=20, dropout_rate=0.1, mc_samples=10):
        super().__init__()
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.t5_encoder = T5EncoderModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.t5_encoder.gradient_checkpointing_enable()
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

class PlannerHead4D(nn.Module):
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        d_model: int = 1024,
        num_waypoints: int = 20,
        adapter_hidden: int = 512,
        num_unfrozen_layers: int = 2,
        dropout_rate: float = 0.1,
        mc_samples: int = 10
    ):
    #def __init__(self, model_name="google/flan-t5-base", d_model=1024, num_waypoints=20, dropout_rate=0.1, mc_samples=10):
        super().__init__()
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.t5_encoder = T5EncoderModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("Loaded Flan-T5-Large in 8-bit mode")
        except (ImportError, RuntimeError) as e:
            print(f"bitsandbytes unavailable ({e}), loading full-precision with offloading")
            self.t5_encoder = T5EncoderModel.from_pretrained(
                model_name,
                device_map="auto",
                offload_folder="offload_dir",     # where to spill tensors on CPU
                offload_state_dict=True,         # offload optimizer & model states
            )
        #bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.num_waypoints = num_waypoints
        self.num_unfrozen_layers = num_unfrozen_layers
        self.mc_samples = mc_samples
        # self.t5_encoder = T5EncoderModel.from_pretrained(
        #     model_name,
        #     quantization_config=bnb_config,
        #     device_map="auto",
        # )
        self._freeze_lower_t5_layers()
        self.t5_encoder.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(dropout_rate)
        # self.regressor = nn.Linear(d_model, num_waypoints * 3)
        # self.training = True
        self.adapter = nn.Sequential(
            nn.Linear(d_model, adapter_hidden),
            nn.ReLU(),
            nn.Linear(adapter_hidden, num_waypoints * 3)
        )
        self.mc_samples = mc_samples

    def _freeze_lower_t5_layers(self):
        # Freeze all parameters
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        # Unfreeze last N encoder blocks
        blocks = self.t5_encoder.encoder.block
        for block in blocks[-self.num_unfrozen_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward_once(self, inputs_embeds):
        enc = self.t5_encoder.encoder(inputs_embeds=inputs_embeds)
        cls_repr = enc.last_hidden_state[:, 0, :]
        dropped = self.dropout(cls_repr)
        preds = self.adapter(dropped)
        return preds.view(preds.size(0), -1, 3)

    def forward(self, bev, time_tokens, pose_emb, route_emb, additional_input=None):
        # Concatenate the 5 inputs (be sure to handle their shapes properly)
        inputs = torch.cat([bev, time_tokens, pose_emb, route_emb], dim=1)
        
        # Add the 5th input if it is provided (additional_input should have shape (B, D, 1024))
        if additional_input is not None:
            inputs = torch.cat([inputs, additional_input], dim=1)

        # If not training and MC samples > 1, use Monte Carlo sampling
        if not self.training and self.mc_samples > 1:
            self.t5_encoder.train()
            samples = [self.forward_once(inputs) for _ in range(self.mc_samples)]
            self.t5_encoder.eval()
            stacked = torch.stack(samples, dim=0)
            return stacked.mean(dim=0), stacked.var(dim=0)
        else:
            pred = self.forward_once(inputs)
            return pred, torch.zeros_like(pred)