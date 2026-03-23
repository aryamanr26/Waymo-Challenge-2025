import os
import warnings

import torch
import torch.optim as optim

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset
from waymo_e2e.losses import ade_fde_loss
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

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    EPOCHS = 1
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        total_loss = 0.0
        for images, _intent, _past, future_states, pose_token, routing_token in train_ds.take(10):
            images = torch.from_numpy(images.numpy()).to(device)
            pose_token = torch.from_numpy(pose_token.numpy()).to(device)
            future_states = torch.from_numpy(future_states.numpy()).to(device)
            routing_token = torch.from_numpy(routing_token.numpy()).to(device)
            images = images.permute(0, 1, 4, 2, 3)

            optimizer.zero_grad()
            waypoints_mean, _ = model(images, pose_token, routing_token)
            loss = ade_fde_loss(waypoints_mean, future_states)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} total loss: {total_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            path = f"e2e_planner_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), path)
            print(f"Saved {path}")

    print("Training completed.")
