import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from loader.torchdata import WaymoTFRecordStreamer
from transformers import CLIPVisionModel

class MonocularDepthHead(nn.Module):
    """
    Lightweight CNN decoder that takes ViT patch features and produces a dense depth map.
    """
    def __init__(self, in_dim=1024):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Final upsample to match input resolution (e.g. 224x224)
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),

            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_feats: (B, P, D), where P is num patches, D is feature dim.
        Returns:
            depth: (B, H, W) depth map.
        """
        B, P, D = patch_feats.shape
        # infer feature map size
        Hf = Wf = int(P ** 0.5)
        # reshape to (B, D, Hf, Wf)
        x = patch_feats.permute(0, 2, 1).reshape(B, D, Hf, Wf)
        depth = self.decoder(x)       # (B, 1, H, W)
        depth = depth.squeeze(1)      # (B, H, W)
        return depth

class PhotometricSSIMLoss(nn.Module):
    """
    Combined SSIM + L1 photometric loss.
    """
    def __init__(self, alpha: float = 0.85):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred:   (B, H, W) predicted image or disparity map
            target: (B, H, W) reference image
        Returns:
            scalar photometric loss
        """
        # SSIM expects (B, C, H, W)
        pred_ = pred.unsqueeze(1)
        target_ = target.unsqueeze(1)
        ssim_val = ssim(pred_, target_, data_range=1.0, size_average=True)
        l1 = torch.abs(pred - target).mean(dim=[1,2])  # (B,)
        photo = self.alpha * (1 - ssim_val) + (1 - self.alpha) * l1
        return photo.mean()

def run_depth_with_waymo(batch_size=4, num_batches=1, device=None):
    # Setup device
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # 1) Stream Waymo TFRecords into torch tensors
    pattern = 'gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training_*.tfrecord*'
    streamer = WaymoTFRecordStreamer(gcs_pattern=pattern, batch_size=batch_size)

    # 2) Instantiate models
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    depth_head = MonocularDepthHead(in_dim=clip_model.config.hidden_size).to(device)
    ssim_loss = PhotometricSSIMLoss(alpha=0.85).to(device)

    # 3) Loop over streamed batches
    for batch_idx, (imgs_tf, pose_tf, route_tf) in enumerate(streamer.dataset):
        if batch_idx >= num_batches:
            break
        # Convert TF tensors to Torch
        imgs = torch.from_numpy(imgs_tf.numpy()).to(device)   # [B, 8, 3, 224, 224]
        pose = torch.from_numpy(pose_tf.numpy()).to(device)   # [B, 32]
        route = torch.from_numpy(route_tf.numpy()).to(device) # [B, 16]
        B, V, C, H, W = imgs.shape

        # Flatten and encode
        imgs_flat = imgs.view(B * V, C, H, W)                # [B*V, 3, 224, 224]
        with torch.no_grad():
            enc_out = clip_model(pixel_values=imgs_flat).last_hidden_state  # [B*V, 1+P, D]
        patch_feats = enc_out[:, 1:, :]                      # [B*V, P, D]

        # Predict depth
        depth_flat = depth_head(patch_feats)                 # [B*V, H, W]
        depth_maps = depth_flat.view(B, V, H, W)             # [B, 8, 224, 224]

        # Example SSIM loss between view0 and view1 RGB reprojections (placeholder)
        # In practice, warp imgs[:,0] -> imgs[:,1] using depth_maps[:,0] & poses, then compare to imgs[:,1]
        d0, d1 = depth_maps[:, 0], depth_maps[:, 1]
        loss = ssim_loss(d0, d1)
        print(f"[Batch {batch_idx}] Depth maps: {depth_maps.shape}, Demo SSIM loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_depth_with_waymo(batch_size=4, num_batches=1)