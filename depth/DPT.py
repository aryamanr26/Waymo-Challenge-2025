import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load model + feature extractor
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)

# Utility function to plot 3 camera images side by side
def plot_front_3_camera_images(left_image, front_image, right_image, title="Front 3 Camera Views", depth=False):
    image_list = [left_image, front_image, right_image]

    # Check that all images have the same height
    heights = [img.shape[0] for img in image_list]
    if len(set(heights)) != 1:
        raise ValueError("All images must have the same height to concatenate horizontally.")

    concatenated_image = np.concatenate(image_list, axis=1)

    plt.imshow(concatenated_image, cmap="inferno" if depth else None)
    plt.axis('off')
    plt.title(title)

# Load Waymo dataset
loader = WaymoDatasetLoader()
train_files, _, _ = loader.get_file_lists()
dataset_builder = WaymoE2EDataset(batch_size=8)
train_ds = dataset_builder.build_dataset(train_files[0])

# Process a single batch
for images, _, _, _, pose_token, routing_token in train_ds.take(1):
    images = torch.from_numpy(images.numpy()).to(device)
    pose_token = torch.from_numpy(pose_token.numpy()).to(device)
    routing_token = torch.from_numpy(routing_token.numpy()).to(device)

# Process and estimate depth for front 3 camera images
image_list, depth_list = [], []

for i in range(3):
    image_tensor = images[0][i]
    image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    
    # Resize the image to 224x224
    image_resized = cv2.resize(image_np, (224, 224))
    image_list.append(image_resized)

    # Prepare input for depth estimation
    inputs = feature_extractor(images=image_resized, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

    print("Depth range:", predicted_depth.min(), predicted_depth.max())
    
    # Resize the predicted depth to 224x224
    predicted_depth_resized = cv2.resize(predicted_depth, (224, 224), interpolation=cv2.INTER_LINEAR)
    depth_list.append(predicted_depth_resized)

# Plotting the images and depth maps
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
