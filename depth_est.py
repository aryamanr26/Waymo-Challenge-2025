import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import DPTFeatureExtractor, DPTImageProcessor, DPTForDepthEstimation
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset


class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-hybrid-midas"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # self.feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
        self.feature_extractor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)

    def estimate_depth_single(self, image_batch):
        """
        Args:
            SINGLE (torch.Tensor or np.ndarray): Batch of shape (B, H, W, 3)

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
            IMAGE_BATCH (torch.Tensor): Tensor of shape (B, V, 3, H, W) in [0, 1] float range.

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

        predicted_depths = torch.from_numpy(np.stack(predicted_depths)).view(B, V, 224, 224).to(self.device)
        return predicted_depths


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
        depth_map = depth_estimator.estimate_depth_batch([image_np])[0].numpy()
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
