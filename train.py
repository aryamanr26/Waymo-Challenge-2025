import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
from accelerate import Accelerator
accelerator = Accelerator()
import sys
import time
import warnings
import numpy as np
import torch
import torch.optim as optim

# Set environment variables
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def save_checkpoint(epoch, model_components, optimizer, loss, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state': {name: comp.state_dict() for name, comp in model_components.items()},
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model_components, optimizer=None):
    checkpoint = torch.load(filename, map_location=device)
    for name, comp in model_components.items():
        comp.load_state_dict(checkpoint['model_state'][name])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch'], checkpoint['loss']

if __name__ == "__main__":
    # Dataset setup
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    dataset_builder = WaymoE2EDataset(batch_size=8)
    train_ds = dataset_builder.build_dataset(train_files)
    INTENT_MAP = WaymoE2EDataset.INTENT_MAP

    # Model setup
    B, V, C, H, W = 8, 8, 3, 224, 224
    M, NM, D = 16, 128, 1024
    bev_dim, bev_h, bev_w = 64, 16, 16

    qformer = MultiViewQFormer(
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
    qformer, encoder, tf_module, pose_encoder, route_encoder, planner, DPT = \
        accelerator.prepare(qformer, encoder, tf_module, pose_encoder, route_encoder, planner, DPT)

    # Optimizer setup
    optimizer = optim.AdamW([
    *qformer.parameters(),
    *encoder.parameters(),
    *tf_module.parameters(),
    *pose_encoder.parameters(),
    *route_encoder.parameters(),
    *planner.parameters(),
    ], lr=1e-4)
    optimizer = accelerator.prepare(optimizer)

    model_components = {
    'qformer': qformer,
    'encoder': encoder,
    'tf_module': tf_module,
    'pose_encoder': pose_encoder,
    'route_encoder': route_encoder,
    'planner': planner,
    'DPT': DPT
    }

    # Resume training option
    resume_from = None  # e.g., "checkpoint_epoch_5.pth"
    start_epoch = 0

    if resume_from:
        start_epoch, last_loss = load_checkpoint(resume_from, model_components, optimizer)
        print(f"Resumed from checkpoint: epoch {start_epoch}, loss={last_loss:.4f}")

    EPOCHS = 10
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        loop_iter = 0

        for images, intent, past_states, future_states, pose_token, routing_token in train_ds: # remove 10 for full training
            images = torch.from_numpy(images.numpy()).to(device)
            pose_token = torch.from_numpy(pose_token.numpy()).to(device)
            future_states = torch.from_numpy(future_states.numpy()).to(device)
            routing_token = torch.from_numpy(routing_token.numpy()).to(device)

            optimizer.zero_grad()

            # Preprocess
            #images = images.permute(0, 1, 4, 2, 3)  # (B, V, C, H, W)
            dimages = images.permute(0, 1, 4, 2, 3)
            # with torch.no_grad():
            all_depths = []
            for i in range(dimages.shape[0]):
                one_sample = dimages[i : i+1]           # shape (1, 8, 224, 224, 3)
                depths_i = DPT.estimate_depth_batch(one_sample)  # (1, 8, 224, 224)
                all_depths.append(depths_i)
            depth_maps = torch.cat(all_depths, dim=0)  # (8, 8, 224, 224)
            depth_maps = depth_maps.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (B, V, 3, 224, 224)
            depth_maps = depth_maps.to(device)
            # Forward pass
            depth_tokens = qformer(depth_maps)
            depth_emb = encoder(depth_tokens)
            images = images.permute(0, 1, 4, 2, 3).to(device)
            visual_tokens = qformer(images)
            bev_emb = encoder(visual_tokens)
            time_embedded = tf_module(visual_tokens)

            pose_emb = pose_encoder(pose_token)
            route_emb = route_encoder(routing_token)
            means, vars_ = [], []
            B_actual = bev_emb.shape[0]
            for i in range(B_actual):
                bev_i   = bev_emb[i : i+1]
                temp_i  = time_embedded[i : i+1]
                pose_i  = pose_emb[i : i+1]
                route_i = route_emb[i : i+1]
                depth_i = depth_emb[i : i+1]

                torch.cuda.empty_cache()
                mean_i, var_i = planner(bev_i, temp_i, pose_i, route_i, depth_i)

                means.append(mean_i)   # (1, N, 2)
                vars_.append(var_i)

            # Re-stack into full batch
            waypoints_mean = torch.cat(means, dim=0)  # (B, N, 2)
            waypoints_var  = torch.cat(vars_,  dim=0)  # (B, N, 2)
            #print("Waypoints mean shape:", waypoints_mean.shape)

            end_time = time.time()
            #print(f"Inference time: {end_time - start_time:.4f} seconds")
            
            # Quick view of the array
            # print("Future States array:", future_states[0])
            # print("Predicted Future States array:", waypoints_mean[0])

            # Printing the current status
            if loop_iter % 50 == 0:
                print(f"Loop has reached {loop_iter+1} with loss: {total_loss}")
            # Compute loss
            loss = ade_fde_loss(waypoints_mean, future_states)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop_iter += 1

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        # Save the model
    #     if (epoch + 1) % 5 == 0:
    #         torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    #         torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
    #         torch.save(tf_module.state_dict(), f"tf_module_epoch_{epoch+1}.pth")
    #         torch.save(pose_encoder.state_dict(), f"pose_encoder_epoch_{epoch+1}.pth")
    #         torch.save(route_encoder.state_dict(), f"route_encoder_epoch_{epoch+1}.pth")
    #         torch.save(planner.state_dict(), f"planner_epoch_{epoch+1}.pth")
    #         torch.save(DPT.state_dict(), f"DPT_epoch_{epoch+1}.pth")
    #         print(f"Model saved for epoch {epoch+1}")
    #     print(f"Epoch {epoch+1} completed.")
    # print("Training completed.")
    # print("Model saved.")




    
        
