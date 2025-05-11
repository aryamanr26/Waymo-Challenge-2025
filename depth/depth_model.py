import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from transformers import CLIPVisionModel
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset
import matplotlib.pyplot as plt

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
    

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()


    INTENT_MAP = WaymoE2EDataset.INTENT_MAP

    dataset_builder = WaymoE2EDataset(batch_size=8)
    train_ds = dataset_builder.build_dataset(train_files[0])

    for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(1):
            images = torch.from_numpy(images.numpy()).to(device)
            print(images.shape)
            # intent = torch.from_numpy(intent.numpy())
            # past_states = torch.from_numpy(past_states.numpy())
            # future_states = torch.from_numpy(future_states.numpy())
            pose_token = torch.from_numpy(pose_token.numpy()).to(device)
            routing_token = torch.from_numpy(routing_token.numpy()).to(device)
    
    predictor = DepthPredictor()
    depth_map = predictor.predict_depth(images.permute(0, 1, 4, 2, 3))

    rgb_img = images[0][0].detach().cpu().numpy()       # (224, 224, 3)
    depth_img = depth_map[0][0].detach().cpu().numpy()  # (224, 224)
    print(depth_map.shape)
    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(rgb_img)
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    axs[1].imshow(depth_img, cmap='inferno')
    axs[1].set_title("Predicted Depth")
    axs[1].axis("off")

    plt.show()
