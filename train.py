import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.optim as optim

# Set environment variables
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# sys.stderr = open(os.devnull, 'w')
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch using:", device)

# Waymo dataset imports
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset

# Model components
from vision_encoder import MultiViewQFormer
from bev import BEVFeatureEncoder
from tfusion import TemporalFusion
from embedding import PoseTokenEncoder, RouteTokenEncoder
from planner_head import PlannerHead4D
from depth_est import DepthEstimator

# Loss function
def ade_fde_loss(predictions, targets, mask=None, alpha=1.0, beta=0.8):
    """
    Args:
        predictions: Tensor of shape (B, T, 3)
        targets: Tensor of shape (B, T, 3)
        mask: Optional boolean tensor of shape (B, T)
        alpha: Weight for ADE
        beta: Weight for FDE
    Returns:
        Combined ADE + FDE loss (scalar)
    """
    # Euclidean distance over (x, y, z)
    l2_dist = torch.norm(predictions - targets, dim=-1)  # (B, T)

    # ADE: average across time
    if mask is not None:
        ade = (l2_dist * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    else:
        ade = l2_dist.mean(dim=-1)

    # FDE: final timestep distance
    final_pred = predictions[:, -1]  # (B, 3)
    final_target = targets[:, -1]    # (B, 3)
    fde = torch.norm(final_pred - final_target, dim=-1)  # (B,)

    return (alpha * ade.mean()) + (beta * fde.mean())

if __name__ == "__main__":
    # Dataset setup
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    dataset_builder = WaymoE2EDataset(batch_size=8)
    train_ds = dataset_builder.build_dataset(train_files[0])
    INTENT_MAP = WaymoE2EDataset.INTENT_MAP

    # Model setup
    B, V, C, H, W = 8, 8, 3, 224, 224
    M, NM, D = 16, 128, 1024
    bev_dim, bev_h, bev_w = 64, 16, 16

    model = MultiViewQFormer(
        num_views=V, num_queries_per_view=M,
        vision_model_name="openai/clip-vit-large-patch14",
        num_layers=2, num_heads=16
    ).to(device)

    encoder = BEVFeatureEncoder(
        num_views=V, num_queries_per_view=M,
        query_dim=D, bev_dim=bev_dim,
        bev_h=bev_h, bev_w=bev_w
    ).to(device)

    tf_module = TemporalFusion(num_positions=NM, d_model=D).to(device)
    pose_encoder = PoseTokenEncoder(input_dim=64, d_model=D).to(device)
    route_encoder = RouteTokenEncoder(input_dim=16, d_model=D).to(device)
    planner = PlannerHead4D(model_name="google/flan-t5-large", d_model=D).to(device)
    DPT = DepthEstimator()

    # Optimizer setup
    optimizer = optim.AdamW([
    *model.parameters(),
    *encoder.parameters(),
    *tf_module.parameters(),
    *pose_encoder.parameters(),
    *route_encoder.parameters(),
    *planner.parameters(),
    ], lr=1e-4)

    EPOCHS = 1
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0

        for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(10): # remove 10 for full training
            images = torch.from_numpy(images.numpy()).to(device)
            pose_token = torch.from_numpy(pose_token.numpy()).to(device)
            future_states = torch.from_numpy(future_states.numpy()).to(device)
            routing_token = torch.from_numpy(routing_token.numpy()).to(device)

            optimizer.zero_grad()

            # Preprocess
            images = images.permute(0, 1, 4, 2, 3)  # (B, V, C, H, W)
            depth_maps = DPT.estimate_depth_batch(images).unsqueeze(2).repeat(1, 1, 3, 1, 1).to(device)

            # Forward pass
            depth_tokens = model(depth_maps)
            depth_emb = encoder(depth_tokens)

            visual_tokens = model(images)
            bev_emb = encoder(visual_tokens)
            time_embedded = tf_module(visual_tokens)

            pose_emb = pose_encoder(pose_token)
            route_emb = route_encoder(routing_token)

            waypoints_mean, waypoints_var = planner(bev_emb, time_embedded, pose_emb, route_emb, depth_emb)

            # Quick view of the array
            # print("Future States array:", future_states[0])
            # print("Predicted Future States array:", waypoints_mean[0])

            # Compute loss
            loss = ade_fde_loss(waypoints_mean, future_states)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        # Save the model
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
            torch.save(tf_module.state_dict(), f"tf_module_epoch_{epoch+1}.pth")
            torch.save(pose_encoder.state_dict(), f"pose_encoder_epoch_{epoch+1}.pth")
            torch.save(route_encoder.state_dict(), f"route_encoder_epoch_{epoch+1}.pth")
            torch.save(planner.state_dict(), f"planner_epoch_{epoch+1}.pth")
            torch.save(DPT.state_dict(), f"DPT_epoch_{epoch+1}.pth")
            print(f"Model saved for epoch {epoch+1}")
        print(f"Epoch {epoch+1} completed.")
    print("Training completed.")
    print("Model saved.")




    
        
