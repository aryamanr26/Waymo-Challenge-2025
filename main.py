import os
import sys
import time
import warnings
import numpy as np
import torch

# Set environment variables
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
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

    # Processing one batch
    for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(1):
        start_time = time.time()

        images = torch.from_numpy(images.numpy()).to(device)
        pose_token = torch.from_numpy(pose_token.numpy()).to(device)
        routing_token = torch.from_numpy(routing_token.numpy()).to(device)

        print("Stacked Image shape:", images.shape)
        # print("Intent:", intent[0].item(), "->", INTENT_MAP.get(int(intent[0]), "Unknown"))
        # print("Past states shape:", past_states[0].shape)
        # print("Future states shape:", future_states[0].shape)
        # print("Pose token shape:", pose_token.shape)
        # print("Routing token shape:", routing_token.shape)

        # Estimate depth for each image in batch
        # depth_maps = [DPT.estimate_depth_batch(images[b]) for b in range(B)]
        # depth_maps = torch.stack(depth_maps).unsqueeze(-1).repeat(1, 1, 1, 1, 3)
        # depth_maps = depth_maps.permute(0, 1, 4, 2, 3).to(device)

        # Estimate depth for each image in batch
        dimages = images.permute(0, 1, 4, 2, 3)
        with torch.no_grad():
            depth_maps = DPT.estimate_feature_batch(dimages)  # (B, V, 224, 224)
            depth_maps = depth_maps.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (B, V, 3, 224, 224)

        depth_maps = depth_maps.to(device)
        print(depth_maps.shape)
        # Depth-based visual encoding
        depth_tokens = model(depth_maps)
        depth_emb = encoder(depth_tokens)
        print("Depth Tokens:", depth_tokens.shape)
        print("Depth Embeddings:", depth_emb.shape)

        # Normal visual encoding
        images = images.permute(0, 1, 4, 2, 3).to(device)
        visual_tokens = model(images)
        bev_emb = encoder(visual_tokens)
        time_embedded = tf_module(visual_tokens)

        print("Visual Tokens:", visual_tokens.shape)
        print("BEV Embeddings:", bev_emb.shape)
        print("Temporal Tokens:", time_embedded.shape)

        pose_emb = pose_encoder(pose_token)
        route_emb = route_encoder(routing_token)

        # Planning
        waypoints_mean, waypoints_var = planner(bev_emb, time_embedded, pose_emb, route_emb, depth_emb)
        print("Waypoints mean shape:", waypoints_mean.shape)

        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f} seconds")
