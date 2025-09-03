import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import DPTFeatureExtractor, DPTImageProcessor, DPTForDepthEstimation
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset

class DepthPatchEncoder(nn.Module):
    def __init__(self, patch_size=16, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Each depth map is (1, 224, 224); treat it like a grayscale image
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, depth_batch):
        """
        Args:
            depth_batch (torch.Tensor): (B, V, 224, 224)
        Returns:
            torch.Tensor: (B, V, N_patches, embed_dim) where N_patches = (224/patch_size)**2 = 14*14 = 196
        """
        B, V, H, W = depth_batch.shape
        assert H == 224 and W == 224, "Expected depth maps of shape (B, V, 224, 224)"
        x = depth_batch.view(B*V, 1, H, W)  # → (BV, 1, 224, 224)

        x = self.proj(x)  # → (B*V, embed_dim, 14, 14) if patch_size = 16
        x = x.flatten(2).transpose(1, 2)  # → (B*V, N_patches, embed_dim)

        x = x.view(B, V, -1, self.embed_dim)  # → (B, V, N_patches, embed_dim)
        return x
    
class DepthFusionEmbedder(nn.Module):
    def __init__(self, patch_size=16, embed_dim=1024, n_tokens=256, add_positional=True, project_to=None):
        super().__init__()
        self.encoder = DepthPatchEncoder(patch_size=patch_size, embed_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.n_tokens = n_tokens

        self.add_positional = add_positional
        if add_positional:
            self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, embed_dim))

        # Optional projection layer to match LLM hidden dim (e.g., 1024 → 768)
        self.proj = nn.Linear(embed_dim, project_to) if project_to else nn.Identity()

    def forward(self, depth_maps):  # depth_maps: (B, V, 224, 224)
        B, V, H, W = depth_maps.shape
        tokens = self.encoder(depth_maps)             # (B, V, 196, 1024)
        tokens = tokens[:, :, :128, :]                # (B, V, 128, 1024)
        tokens = self.norm(tokens)                    # (B, V, 128, 1024)
        tokens = tokens.view(B * V, 128, -1)           # (B', 128, 1024)

        # Expand to 256 tokens
        tokens = tokens.repeat(1, 2, 1)[:, :self.n_tokens, :]  # (B', 256, 1024)

        # ======= Add this block to reduce batch and tokens =======
        tokens = tokens.view(B, V, self.n_tokens, -1)   # (B=8, V=8, 256, 1024)
        tokens = tokens.mean(dim=1)                      # average over views → (8, 256, 1024)

        tokens = tokens.transpose(1, 2)                  # (8, 1024, 256)
        tokens = torch.nn.functional.avg_pool1d(tokens, kernel_size=2, stride=2)  # pool tokens: (8, 1024, 128)
        tokens = tokens.transpose(1, 2)                  # (8, 128, 1024)
        # ============================================================
        if self.add_positional:
            tokens = tokens + self.pos_embed[:, :tokens.size(1), :]  # add positional embedding
            
        return self.proj(tokens)                         # (B', 256, project_to or 1024)
    
class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-hybrid-midas", compile_model=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # self.feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
        self.feature_extractor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
        
        if compile_model and torch.__version__ >= "2.0" and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model)
                print("✅ torch.compile() applied successfully")
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")

    def estimate_depth_single(self, image_batch):
        """
        Args:
            image_batch (torch.Tensor or np.ndarray): Batch of shape (B, H, W, 3)

        Returns:
            torch.Tensor: Predicted depth maps of shape (B, 224, 224)
        """
        if isinstance(image_batch, torch.Tensor):
            image_batch = image_batch.detach().cpu().numpy()

        if image_batch.dtype != np.uint8:
            image_batch = (image_batch * 255).astype(np.uint8)

        resized_images = [cv2.resize(img, (224, 224)) for img in image_batch]
        inputs = self.feature_extractor(images=resized_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depths = outputs.predicted_depth.squeeze()  # shape: (B, H, W)

        if predicted_depths.ndim == 2:  # Handle single image case
            predicted_depths = predicted_depths.unsqueeze(0)

        predicted_depths = [
            cv2.resize(depth.cpu().numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)
            for depth in predicted_depths
        ]

        predicted_depths = torch.from_numpy(np.stack(predicted_depths))
        return predicted_depths  # shape: (B, 224, 224)
    
    def estimate_depth_batch(self, image_batch):
        """
        Args:
            image_batch (torch.Tensor): Tensor of shape (B, V, 3, H, W) in [0, 1] float range.

        Returns:
            torch.Tensor: Predicted depth maps of shape (B, V, 224, 224)
        """
        B, V, C, H, W = image_batch.shape

        # Move to CPU + convert to numpy
        image_batch = image_batch.detach().cpu().numpy()

        # Convert to uint8
        image_batch = (image_batch * 255).astype(np.uint8)

        # Transpose to (B*V, H, W, 3) for processor
        image_batch = image_batch.transpose(0, 1, 3, 4, 2).reshape(B * V, H, W, 3)

        resized_images = [cv2.resize(img, (224, 224)) for img in image_batch]
        inputs = self.feature_extractor(images=resized_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depths = outputs.predicted_depth.squeeze()  # (B*V, H, W)

        if predicted_depths.ndim == 2:
            predicted_depths = predicted_depths.unsqueeze(0)

        # Resize all depth maps back to (224, 224)
        predicted_depths = [
            cv2.resize(depth.cpu().numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)
            for depth in predicted_depths
        ]

        predicted_depths = torch.from_numpy(np.stack(predicted_depths)).view(B, V, 224, 224)
        return predicted_depths

    def estimate_depth_batch_torch(self, image_batch):
  
        """
        Args:
            image_batch (torch.Tensor): shape (B, V, 224, 224, 3), values in [0, 1]

        Returns:
            torch.Tensor: shape (B, V, 224, 224)
        """
        
        B, V, H, W, C = image_batch.shape
        assert C == 3, "Expected 3 channels (RGB)"
        assert image_batch.dtype == torch.float32

        # Flatten to a list of individual images in (H, W, C) format
        image_list = image_batch.reshape(B * V, H, W, C)  # shape: (B*V, H, W, C)
        image_list = [img for img in image_list]  # convert to list of tensors

        # Pass list to feature extractor
        inputs = self.feature_extractor(
            images=image_list,
            return_tensors="pt",
            input_data_format="channels_last",
            do_rescale=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depths = outputs.predicted_depth  # shape: (B*V, h, w)

        # Resize output to (224, 224) if needed
        resized_depths = torch.nn.functional.interpolate(
            predicted_depths.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(1)  # shape: (B*V, 224, 224)

        return resized_depths.view(B, V, 224, 224)

def plot_front_3_camera_images(left_image, front_image, right_image, title="Front 3 Camera Views", depth=False):
    image_list = [left_image, front_image, right_image]
    heights = [img.shape[0] for img in image_list]
    if len(set(heights)) != 1:
        raise ValueError("All images must have the same height to concatenate horizontally.")

    concatenated_image = np.concatenate(image_list, axis=1)
    plt.imshow(concatenated_image, cmap="inferno" if depth else None)
    plt.axis('off')
    plt.title(title)


def load_waymo_batch(batch_size=8):
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    dataset_builder = WaymoE2EDataset(batch_size=batch_size)
    train_ds = dataset_builder.build_dataset(train_files[0])

    for images, _, _, _, pose_token, routing_token in train_ds.take(1):
        images = torch.from_numpy(images.numpy())
        pose_token = torch.from_numpy(pose_token.numpy())
        routing_token = torch.from_numpy(routing_token.numpy())
        return images, pose_token, routing_token

    raise ValueError("Failed to load Waymo data batch.")


if __name__ == "__main__":
    depth_estimator = DepthEstimator()

    # Load batch of Waymo images
    images, _, _ = load_waymo_batch()
    image_list, depth_list = [], []

    # Process first sample's front 3 cameras
    for i in range(3):
        image_tensor = images[0][i]
        image_np = (image_tensor.numpy() * 255).astype(np.uint8)
        depth_map = depth_estimator.estimate_depth_single([image_np])[0].numpy()
        image_list.append(cv2.resize(image_np, (224, 224)))
        depth_list.append(depth_map)

    # Plot input images and depth predictions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_front_3_camera_images(*image_list, title="Input Images")

    plt.subplot(1, 2, 2)
    plot_front_3_camera_images(*depth_list, title="Predicted Depth Maps", depth=True)

    plt.tight_layout()
    plt.show()
