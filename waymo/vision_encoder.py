# # import torch
# # import torch.nn as nn
# # from transformers import CLIPVisionModel
# # import tensorflow as tf
# # import math

# # class MultiViewQFormer(nn.Module):
# #     def __init__(
# #         self,
# #         num_views: int = 8,
# #         num_queries_per_view: int = 16,
# #         vision_model_name: str = "openai/clip-vit-large-patch14",
# #         num_layers: int = 2,
# #         num_heads: int | None = None,
# #         d_ff: int | None = None,
# #     ):
# #         super().__init__()
# #         self.num_views = num_views
# #         # 1) Load & freeze vision encoder
# #         self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
# #         for p in self.vision_encoder.parameters():
# #             p.requires_grad = False
# #         # Hidden dimension from model config
# #         self.d_model = self.vision_encoder.config.hidden_size
# #         # Number of heads: default to model config if not specified
# #         if num_heads is None:
# #             num_heads = self.vision_encoder.config.num_attention_heads
# #         # Feed-forward dimension: default to 4x hidden size
# #         if d_ff is None:
# #             d_ff = 4 * self.d_model
# #         # 2) Camera-ID embeddings
# #         self.camera_id_emb = nn.Embedding(num_views, self.d_model)
# #         # 3) Learnable queries
# #         self.num_queries = num_views * num_queries_per_view
# #         self.queries = nn.Parameter(torch.randn(self.num_queries, self.d_model) * 0.02)
# #         # 4) Build Q-Former as a TransformerDecoder
# #         decoder_layer = nn.TransformerDecoderLayer(
# #             d_model=self.d_model,
# #             nhead=num_heads,
# #             dim_feedforward=d_ff,
# #             dropout=0.1,
# #             activation="gelu",
# #             batch_first=False,  # expects (seq_len, batch, d_model)
# #         )
# #         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

# #     def forward(self, images: torch.Tensor):
# #         # images: (B, V, 3, H, W)
# #         B, V, C, H, W = images.shape
# #         x = images.view(B * V, C, H, W)
# #         # Extract patch embeddings
# #         out = self.vision_encoder(pixel_values=x).last_hidden_state  # (B*V, 1+P, d_model)
# #         patches = out[:, 1:, :]  # discard [CLS] token -> (B*V, P, d_model)
# #         P = patches.size(1)
# #         # Reshape and add camera ID embeddings
# #         patches = patches.view(B, V, P, self.d_model)
# #         cam_ids = torch.arange(V, device=images.device)
# #         cam_emb = self.camera_id_emb(cam_ids)  # (V, d_model)
# #         patches = patches + cam_emb.unsqueeze(0).unsqueeze(2)
# #         # Flatten views into memory sequence for cross-attention
# #         memory = patches.view(B, V * P, self.d_model).permute(1, 0, 2)  # (V*P, B, d_model)
# #         # Prepare queries: (Q, B, d_model)
# #         queries = self.queries.unsqueeze(1).expand(-1, B, -1)
# #         # Decode: queries attend to multi-view memory
# #         q_out = self.decoder(tgt=queries, memory=memory)
# #         # Return (B, Q, d_model)
# #         return q_out.permute(1, 0, 2)


# # # Example usage:
# # if __name__ == "__main__":
# #      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #      B, V, C, H, W = 4, 8, 3, 224, 224
# #      M = 16  # queries per view

# #      model = MultiViewQFormer(
# #          num_views=V,
# #          num_queries_per_view=M,
# #          vision_model_name="openai/clip-vit-large-patch14",
# #          num_layers=2,
# #          num_heads=16,  # match ViT-L’s 16 attention heads
# #      ).to(device)

# #      images = torch.randn(B, V, C, H, W, device=device)
# #      visual_tokens = model(images)  # (B, V*M, d_model)
# #      print(visual_tokens.shape)  # e.g., (4, 8*16, 1024) for ViT-L/14

# import torch
# import torch.nn as nn
# from transformers import CLIPVisionModel, CLIPVisionConfig

# class MultiViewQFormer(nn.Module):
#     def __init__(
#         self,
#         num_views: int = 8,
#         num_queries_per_view: int = 16,
#         vision_model_name: str = "openai/clip-vit-large-patch14",
#         num_layers: int = 2,
#         num_heads: int | None = None,
#         d_ff: int | None = None,
#         cache_dir: str | None = None,
#         local_files_only: bool = True,
#     ):
#         super().__init__()
#         self.num_views = num_views
#         # 1) Load & freeze vision encoder config
#         try:
#             config = CLIPVisionConfig.from_pretrained(
#                 vision_model_name,
#                 cache_dir=cache_dir,
#                 local_files_only=local_files_only
#             )
#         except Exception:
#             config = CLIPVisionConfig.from_pretrained(
#                 vision_model_name,
#                 cache_dir=cache_dir,
#                 local_files_only=False
#             )
#         # 2) Load model, try offline first
#         try:
#             self.vision_encoder = CLIPVisionModel.from_pretrained(
#                 vision_model_name,
#                 config=config,
#                 cache_dir=cache_dir,
#                 local_files_only=local_files_only
#             , low_cpu_mem_usage=True)
#         except Exception:
#             self.vision_encoder = CLIPVisionModel.from_pretrained(
#                 vision_model_name,
#                 config=config,
#                 cache_dir=cache_dir,
#                 local_files_only=False
#             )
#         for p in self.vision_encoder.parameters():
#             p.requires_grad = False

#         # Hidden dimension from model config
#         self.d_model = self.vision_encoder.config.hidden_size
#         # Number of heads: default to model config if not specified
#         if num_heads is None:
#             num_heads = self.vision_encoder.config.num_attention_heads
#         # Feed-forward dimension: default to 4x hidden size
#         if d_ff is None:
#             d_ff = 4 * self.d_model

#         # 3) Camera-ID embeddings
#         self.camera_id_emb = nn.Embedding(num_views, self.d_model)
#         # 4) Learnable queries
#         self.num_queries = num_views * num_queries_per_view
#         self.queries = nn.Parameter(torch.randn(self.num_queries, self.d_model) * 0.02)
#         # 5) Build Q-Former as a TransformerDecoder
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=self.d_model,
#             nhead=num_heads,
#             dim_feedforward=d_ff,
#             dropout=0.1,
#             activation="gelu",
#             batch_first=False,
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

#     def forward(self, images: torch.Tensor):
#         # images: (B, V, 3, H, W)
#         B, V, C, H, W = images.shape
#         x = images.view(B * V, C, H, W)
#         # Extract patch embeddings
#         out = self.vision_encoder(pixel_values=x).last_hidden_state
#         patches = out[:, 1:, :]
#         P = patches.size(1)
#         # Reshape and add camera ID embeddings
#         patches = patches.view(B, V, P, self.d_model)
#         cam_ids = torch.arange(V, device=images.device)
#         cam_emb = self.camera_id_emb(cam_ids)
#         patches = patches + cam_emb.unsqueeze(0).unsqueeze(2)
#         # Flatten views into memory sequence for cross-attention
#         memory = patches.view(B, V * P, self.d_model).permute(1, 0, 2)
#         # Prepare queries: (Q, B, d_model)
#         queries = self.queries.unsqueeze(1).expand(-1, B, -1)
#         # Decode: queries attend to multi-view memory
#         q_out = self.decoder(tgt=queries, memory=memory)
#         # Return (B, Q, d_model)
#         return q_out.permute(1, 0, 2)

# # Example usage:
# if __name__ == "__main__":
#     import os
#     # Allow for online fallback if offline fails
#     os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     B, V, C, H, W = 4, 8, 3, 224, 224
#     M = 16

#     model = MultiViewQFormer(
#         num_views=V,
#         num_queries_per_view=M,
#         vision_model_name="openai/clip-vit-large-patch14",
#         num_layers=2,
#         num_heads=16,
#         cache_dir=os.path.expanduser("~/.cache/huggingface/transformers"),
#         local_files_only=True
#     ).to(device)

#     images = torch.randn(B, V, C, H, W, device=device)
#     visual_tokens = model(images)
#     print(visual_tokens.shape)  # e.g., (4, 8*16, config.hidden_size)
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig

# Enable CuDNN auto-tuning and PyTorch 2.0 compile for faster kernels
torch.backends.cudnn.benchmark = True
try:
    # PyTorch 2.0+ compilation
    torch._dynamo.config.suppress_errors = True
    compile_model = torch.compile
except AttributeError:
    compile_model = lambda m: m

class MultiViewQFormer(nn.Module):
    def __init__(
        self,
        num_views: int = 8,
        num_queries_per_view: int = 16,
        vision_model_name: str = "openai/clip-vit-large-patch14",
        num_layers: int = 2,
        num_heads: int | None = None,
        d_ff: int | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        # Load config and model with low_cpu_mem_usage and caching
        try:
            config = CLIPVisionConfig.from_pretrained(
                vision_model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only
            )
        except Exception:
            config = CLIPVisionConfig.from_pretrained(
                vision_model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
        try:
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                vision_model_name,
                config=config,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                
            )
        except Exception:
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                vision_model_name,
                config=config,
                cache_dir=cache_dir,
                local_files_only=False,
                
            )
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.d_model = self.vision_encoder.config.hidden_size
        if num_heads is None:
            num_heads = self.vision_encoder.config.num_attention_heads
        if d_ff is None:
            d_ff = 4 * self.d_model

        self.camera_id_emb = nn.Embedding(num_views, self.d_model)
        self.num_queries = num_views * num_queries_per_view
        self.queries = nn.Parameter(torch.randn(self.num_queries, self.d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation="gelu",
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, images: torch.Tensor):
        # images: (B, V, 3, H, W)
        B, V, C, H, W = images.shape
        x = images.view(B * V, C, H, W)
        # Extract patch embeddings
        out = self.vision_encoder(pixel_values=x).last_hidden_state
        patches = out[:, 1:, :]
        P = patches.size(1)

        # Add camera ID embeddings
        patches = patches.view(B, V, P, self.d_model)
        cam_emb = self.camera_id_emb(torch.arange(V, device=images.device))
        patches = patches + cam_emb.view(1, V, 1, self.d_model)

        # Prepare memory and queries
        memory = patches.view(B, V * P, self.d_model).permute(1, 0, 2)
        queries = self.queries.unsqueeze(1).expand(-1, B, -1)

        # Transformer decoding
        q_out = self.decoder(tgt=queries, memory=memory)
        return q_out.permute(1, 0, 2)

# Example usage with mixed precision and compiled model
def main():
    import os
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, V, C, H, W = 4, 8, 3, 224, 224
    M = 16

    model = MultiViewQFormer(
        num_views=V,
        num_queries_per_view=M,
        vision_model_name="openai/clip-vit-large-patch14",
        num_layers=2,
        num_heads=16,
        cache_dir=os.path.expanduser("~/.cache/huggingface/transformers"),
        local_files_only=True
    )
    model.to(device)
    model = compile_model(model)

    # Synthetic data
    images = torch.randn(B, V, C, H, W, device=device)

    # Mixed precision inference
    with torch.cuda.amp.autocast():
        visual_tokens = model(images)
    print(visual_tokens.shape)  # Expect (B, V*M, hidden)

if __name__ == "__main__":
    main()
