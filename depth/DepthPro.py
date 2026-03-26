import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    DPTFeatureExtractor, DPTForDepthEstimation,
    DepthProImageProcessorFast, DepthProForDepthEstimation
)
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_e2e.data import WaymoDatasetLoader, WaymoE2EDataset

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load DepthPro model
image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

# Utility to plot camera images side-by-side
def plot_front_3_camera_images(left_image, front_image, right_image, title="Front 3 Camera Views", depth=False):
    image_list = [left_image, front_image, right_image]
    heights = [img.shape[0] for img in image_list]

    if len(set(heights)) != 1:
        raise ValueError("All images must have the same height to concatenate horizontally.")

    concatenated_image = np.concatenate(image_list, axis=1)
    plt.imshow(concatenated_image, cmap="plasma" if depth else None)
    plt.axis('off')
    plt.title(title)

# Load Waymo dataset
loader = WaymoDatasetLoader()
train_files, _, _ = loader.get_file_lists()

INTENT_MAP = WaymoE2EDataset.INTENT_MAP
dataset_builder = WaymoE2EDataset(batch_size=8)
train_ds = dataset_builder.build_dataset(train_files[0])

# Process a single batch
for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(1):
    images = torch.from_numpy(images.numpy()).to(device)
    pose_token = torch.from_numpy(pose_token.numpy()).to(device)
    routing_token = torch.from_numpy(routing_token.numpy()).to(device)

# Process and estimate depth for front 3 camera images
image_list, depth_list = [], []

for i in range(3):
    image_tensor = images[0][i]
    image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)

    inputs = image_processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed = image_processor.post_process_depth_estimation(
        outputs, target_sizes=[(image_pil.height, image_pil.width)]
    )

    predicted_depth = post_processed[0]["predicted_depth"].squeeze()
    normalized_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min() + 1e-8)
    depth_np = (normalized_depth * 255).cpu().numpy().astype("uint8")
    depth_resized = cv2.resize(depth_np, (224, 224), interpolation=cv2.INTER_LINEAR)

    image_list.append(image_np)
    depth_list.append(depth_resized)

    print(f"Depth range: {predicted_depth.min().item():.4f} to {predicted_depth.max().item():.4f}")
    print(f"Depth shape: {depth_resized.shape}")

# Plotting
# Plotting
plt.figure(figsize=(12, 6))

# Left plot: Input Images
plt.subplot(1, 2, 1)
plot_front_3_camera_images(image_list[0], image_list[1], image_list[2], title="Input Images")

# Right plot: Predicted Depth Maps with Colorbar
plt.subplot(1, 2, 2)
plot_front_3_camera_images(depth_list[0], depth_list[1], depth_list[2], title="Predicted Depth Maps", depth=True)

# Add colorbar to the depth plot
# plt.colorbar(im, ax=plt.gca(), orientation='vertical', shrink=0.8, label='Depth Value')

plt.tight_layout()
plt.show()

