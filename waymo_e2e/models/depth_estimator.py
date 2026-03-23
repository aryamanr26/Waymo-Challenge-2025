import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

from waymo_e2e.data.e2e_tf_dataset import WaymoE2EDataset
from waymo_e2e.data.gcs_loader import WaymoDatasetLoader


class DepthEstimator:
    """Frozen DPT depth (optional branch); keep no_grad in training for speed."""

    def __init__(self, model_name="Intel/dpt-hybrid-midas"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.feature_extractor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)

    def estimate_depth_single(self, image_batch):
        if isinstance(image_batch, torch.Tensor):
            image_batch = image_batch.detach().cpu().numpy()
        if image_batch.dtype != np.uint8:
            image_batch = (image_batch * 255).astype(np.uint8)
        resized_images = [cv2.resize(img, (224, 224)) for img in image_batch]
        inputs = self.feature_extractor(images=resized_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depths = outputs.predicted_depth.squeeze()
        if predicted_depths.ndim == 2:
            predicted_depths = predicted_depths.unsqueeze(0)
        predicted_depths = [
            cv2.resize(depth.cpu().numpy(), (224, 224), interpolation=cv2.INTER_LINEAR) for depth in predicted_depths
        ]
        return torch.from_numpy(np.stack(predicted_depths))

    def estimate_depth_batch(self, image_batch):
        """image_batch: (B, V, 3, H, W) float [0,1]. Returns (B, V, 224, 224)."""
        B, V, C, H, W = image_batch.shape
        image_batch = image_batch.detach().cpu().numpy()
        image_batch = (image_batch * 255).astype(np.uint8)
        image_batch = image_batch.transpose(0, 1, 3, 4, 2).reshape(B * V, H, W, 3)
        resized_images = [cv2.resize(img, (224, 224)) for img in image_batch]
        inputs = self.feature_extractor(images=resized_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depths = outputs.predicted_depth.squeeze()
        if predicted_depths.ndim == 2:
            predicted_depths = predicted_depths.unsqueeze(0)
        predicted_depths = [
            cv2.resize(depth.cpu().numpy(), (224, 224), interpolation=cv2.INTER_LINEAR) for depth in predicted_depths
        ]
        return torch.from_numpy(np.stack(predicted_depths)).view(B, V, 224, 224).to(self.device)


def plot_front_3_camera_images(left_image, front_image, right_image, title="Front 3 Camera Views", depth=False):
    image_list = [left_image, front_image, right_image]
    heights = [img.shape[0] for img in image_list]
    if len(set(heights)) != 1:
        raise ValueError("All images must have the same height to concatenate horizontally.")
    concatenated_image = np.concatenate(image_list, axis=1)
    plt.imshow(concatenated_image, cmap="inferno" if depth else None)
    plt.axis("off")
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
