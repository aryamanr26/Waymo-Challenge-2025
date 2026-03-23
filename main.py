import os
import time
import warnings

import torch

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset
from waymo_e2e.pipeline import E2EConfig, E2EVisualPlanner

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch using:", device)

    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    dataset_builder = WaymoE2EDataset(batch_size=8)
    train_ds = dataset_builder.build_dataset(train_files[0])

    cfg = E2EConfig(
        num_views=3,
        num_queries_per_view=16,
        d_model=1024,
        bev_dim=64,
        bev_h=16,
        bev_w=16,
        vision_backbone="clip",
        vision_model_name="openai/clip-vit-base-patch16",
        use_thin_depth_fusion=True,
        use_legacy_dpt_bev_branch=False,
    )
    model = E2EVisualPlanner(cfg).to(device)

    for images, _intent, _past, future_states, pose_token, routing_token in train_ds.take(1):
        t0 = time.time()
        images = torch.from_numpy(images.numpy()).to(device)
        pose_token = torch.from_numpy(pose_token.numpy()).to(device)
        future_states = torch.from_numpy(future_states.numpy()).to(device)
        routing_token = torch.from_numpy(routing_token.numpy()).to(device)
        images = images.permute(0, 1, 4, 2, 3)

        waypoints_mean, waypoints_var = model(images, pose_token, routing_token)
        print("Stacked Image shape:", images.shape)
        print("Waypoints mean shape:", waypoints_mean.shape)
        print(f"Inference time: {time.time() - t0:.4f}s")
        print("Future states:", future_states[0])
        print("Predicted:", waypoints_mean[0])
