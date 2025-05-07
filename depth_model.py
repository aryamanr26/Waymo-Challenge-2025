import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from transformers import CLIPVisionModel

class MonocularDepthHead(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_dim, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        B, P, D = patch_feats.shape
        Hf = Wf = int(P ** 0.5)
        x = patch_feats.permute(0, 2, 1).reshape(B, D, Hf, Wf)
        depth = self.decoder(x)
        return depth.squeeze(1)

class PhotometricSSIMLoss(nn.Module):
    def __init__(self, alpha: float = 0.85):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred_ = pred.unsqueeze(1)
        target_ = target.unsqueeze(1)
        ssim_val = ssim(pred_, target_, data_range=1.0, size_average=True)
        l1 = torch.abs(pred - target).mean(dim=[1, 2])
        return (self.alpha * (1 - ssim_val) + (1 - self.alpha) * l1).mean()

class DepthPredictor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.depth_head = MonocularDepthHead(in_dim=self.clip_model.config.hidden_size).to(self.device)
        self.ssim_loss_fn = PhotometricSSIMLoss(alpha=0.85).to(self.device)

    def predict_depth(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: torch.Tensor of shape (B, 3, 224, 224) or (B, V, 3, 224, 224)
        Returns:
            depth_maps: (B, 224, 224) or (B, V, 224, 224)
        """
        if images.dim() == 5:
            B, V, C, H, W = images.shape
            images_flat = images.view(B * V, C, H, W)
        else:
            images_flat = images
            B, C, H, W = images.shape
            V = 1

        with torch.no_grad():
            enc_out = self.clip_model(pixel_values=images_flat).last_hidden_state
        patch_feats = enc_out[:, 1:, :]
        depth_flat = self.depth_head(patch_feats)
        if V == 1:
            return depth_flat
        return depth_flat.view(B, V, H, W)

    def photometric_loss(self, depth0, depth1):
        return self.ssim_loss_fn(depth0, depth1)
