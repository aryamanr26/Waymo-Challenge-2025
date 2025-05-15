# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
# from waymo_dataset_loader import WaymoDatasetLoader
# from waymo_E2EDataset import WaymoE2EDataset
# from vision_encoder import MultiViewQFormer
# from bev import BEVFeatureEncoder
# from tfusion import TemporalFusion
# from depth_model import DepthPredictor
# from embedding import PoseTokenEncoder, RouteTokenEncoder
# from planner_head import PlannerHead3D
# # tf.config.experimental.set_visible_devices([], 'GPU')  # Disable GPU for TF

# if __name__ == "__main__":
#     loader = WaymoDatasetLoader()
#     train_files, _, _ = loader.get_file_lists()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Print device", device)

#     INTENT_MAP = WaymoE2EDataset.INTENT_MAP

#     B, V, C, H, W = 2, 8, 3, 224, 224  # batch, views, channels, height, width
#     M = 16                             # queries per view
#     NM, D = 128, 1024                  # temporal tokens, hidden dimension
#     bev_dim, bev_h, bev_w = 64, 16, 16
#     dataset_builder = WaymoE2EDataset(batch_size=B)
#     train_ds = dataset_builder.build_dataset(train_files[0])
#     model = MultiViewQFormer(num_views=V,
#         num_queries_per_view=M,
#         vision_model_name="openai/clip-vit-large-patch14",
#         num_layers=2,
#         num_heads=16  # CLIP ViT-L uses 16 heads
#     ).to(device)

#     encoder = BEVFeatureEncoder(
#         num_views=V,
#         num_queries_per_view=M,
#         query_dim=D,
#         bev_dim=bev_dim,
#         bev_h=bev_h,
#         bev_w=bev_w,
#     ).to(device)

#     tf_module = TemporalFusion(
#         num_positions=NM,
#         d_model=D
#     ).to(device)


#     pose_encoder = PoseTokenEncoder(input_dim=64, d_model=D).to(device)
#     route_encoder = RouteTokenEncoder(input_dim=16, d_model=D).to(device)


#     predictor = DepthPredictor()
#     planner = PlannerHead3D(
#         model_name="google/flan-t5-large",
#         d_model=D
#     ).to(device)

#     i = 0


#     for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(2):
#         images = torch.from_numpy(images.numpy()).to(device)
#         #intent = torch.from_numpy(intent.numpy()).to(device)
#         #past_states = torch.from_numpy(past_states.numpy()).to(device)
#         #future_states = torch.from_numpy(future_states.numpy()).to(device)
#         pose_token = torch.from_numpy(pose_token.numpy()).to(device)
#         routing_token = torch.from_numpy(routing_token.numpy()).to(device)

#         print("Stacked Image shape:", images.shape)
#         print("Intent:", intent[0].item(), "->", INTENT_MAP.get(int(intent[0]), "Unknown"))
#         print("Past states shape:", past_states[0].shape)
#         print("Future states shape:", future_states[0].shape)
#         print("Pose token shape:", pose_token.shape)
#         print("Routing token shape:", routing_token.shape)
    
#         modified_image = images.permute(0, 1, 4, 2, 3).to(device)
#         print(modified_image.shape)

#         visual_tokens = model(modified_image).to(device)  # (B, V*M, d_model)
#         print(visual_tokens.shape)  # e.g., (4, 8*16, 1024) for ViT-L/14

    
#         bev_emb = encoder(visual_tokens).to(device)
#         print("Final BEV tokens:", bev_emb.shape)

    
#         time_embedded = tf_module(visual_tokens).to(device)
#         print("Final Temporal tokens:", time_embedded.shape)

#         img_tensor = modified_image[0].to(device)
    
#         print("Image Tensor Shape: ", img_tensor.shape)
#         depth_map = predictor.predict_depth(img_tensor).to(device)
#         print("Predicted depth shape:", depth_map.shape)

#         # Optional: compute loss between two depths (demo)
#         dummy_loss = predictor.photometric_loss(depth_map, depth_map * 0.95).to(device)
#         print(f"SSIM-based photometric loss: {dummy_loss.item():.7f}")

#         pose_emb = pose_encoder(pose_token).to(device)
#         route_emb = route_encoder(routing_token).to(device)
#         waypoints_mean, waypoints_var = planner(bev_emb, time_embedded, pose_emb, route_emb)
#         i += 1

#         print("Waypoints for {} image -> mean:".format(i), waypoints_mean.shape)
#         print("Waypoints for {} image -> variance:".format(i), waypoints_var.shape)
# import os
# import sys

# from waymo_E2EDataset import WaymoE2EDataset
# from vision_encoder import MultiViewQFormer
# from bev import BEVFeatureEncoder
# from tfusion import TemporalFusion
# from depth_model import DepthPredictor
# from embedding import PoseTokenEncoder, RouteTokenEncoder
# from planner_head import PlannerHead3D
# from transformers import CLIPVisionModel, CLIPVisionConfig

# import torch
# from transformers import CLIPVisionModel, CLIPVisionConfig
# import tensorflow as tf
# from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
# from waymo_dataset_loader import WaymoDatasetLoader

# def main():
#     # PyTorch performance tweaks
#     torch.backends.cudnn.benchmark = True
#     try:
#         # PyTorch 2.0 compile
#         torch._dynamo.config.suppress_errors = True
#         compile_model = torch.compile
#     except AttributeError:
#         compile_model = lambda m: m

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Data loader and dataset
#     loader = WaymoDatasetLoader()
#     train_files, _, _ = loader.get_file_lists()
#     dataset_builder = WaymoE2EDataset(batch_size=2)
#     # Build with autotune and prefetch for TF
#     train_ds = (dataset_builder.build_dataset(train_files[0])
#                 .prefetch(tf.data.AUTOTUNE))

#     # Model instantiation and compilation
#     model = MultiViewQFormer(
#         num_views=8,
#         num_queries_per_view=16,
#         vision_model_name="openai/clip-vit-large-patch14",
#         num_layers=2,
#         num_heads=16
#     ).to(device)
#     model = compile_model(model)

#     encoder = BEVFeatureEncoder(num_views=8, num_queries_per_view=16,
#                                 query_dim=1024, bev_dim=64, bev_h=16, bev_w=16).to(device)
#     tf_module = TemporalFusion(num_positions=128, d_model=1024).to(device)
#     pose_encoder = PoseTokenEncoder(input_dim=64, d_model=1024).to(device)
#     route_encoder = RouteTokenEncoder(input_dim=16, d_model=1024).to(device)
#     predictor = DepthPredictor()
#     planner = PlannerHead3D(model_name="google/flan-t5-large", d_model=1024).to(device)

#     # Mixed-precision context for inference
#     for i, data in enumerate(train_ds.take(2), 1):
#         images, intent, past_states, future_states, pose_token, routing_token = data
#         images = torch.from_numpy(images.numpy()).to(device, non_blocking=True)
#         pose_token = torch.from_numpy(pose_token.numpy()).to(device, non_blocking=True)
#         routing_token = torch.from_numpy(routing_token.numpy()).to(device, non_blocking=True)

#         modified_image = images.permute(0,1,4,2,3)

#         # Inference under autocast
#         with torch.cuda.amp.autocast():
#             visual_tokens = model(modified_image)
#             bev_emb = encoder(visual_tokens)
#             time_embedded = tf_module(visual_tokens)
#             depth_map = predictor.predict_depth(modified_image[0])
#             dummy_loss = predictor.photometric_loss(depth_map, depth_map*0.95)
#             pose_emb = pose_encoder(pose_token)
#             route_emb = route_encoder(routing_token)
#             waypoints_mean, waypoints_var = planner(bev_emb, time_embedded, pose_emb, route_emb)

#         # Print shapes
#         print(f"Iteration {i}:")
#         print("Visual tokens:", visual_tokens.shape)
#         print("BEV tokens:", bev_emb.shape)
#         print("Temporal tokens:", time_embedded.shape)
#         print("Depth map:", depth_map.shape)
#         print(f"Photometric loss: {dummy_loss.item():.7f}")
#         print("Waypoints mean:", waypoints_mean.shape)
#         print("Waypoints var:", waypoints_var.shape)

# if __name__ == "__main__":
#     main()

# import os
# import sys
# import time
# import warnings
# import numpy as np
# import torch

# # Set environment variables
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# #sys.stderr = open(os.devnull, 'w')
# #warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# # Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Torch using:", device)

# # Waymo dataset imports
# from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
# from waymo_dataset_loader import WaymoDatasetLoader
# from waymo_E2EDataset import WaymoE2EDataset

# # Model components
# from vision_encoder import MultiViewQFormer
# from bev import BEVFeatureEncoder
# from tfusion import TemporalFusion
# from embedding import PoseTokenEncoder, RouteTokenEncoder
# from planner_head import PlannerHead4D
# from depth_est import DepthEstimator

# if __name__ == "__main__":
#     # Dataset setup
#     loader = WaymoDatasetLoader()
#     train_files, _, _ = loader.get_file_lists()
#     dataset_builder = WaymoE2EDataset(batch_size=8)
#     train_ds = dataset_builder.build_dataset(train_files[0])
#     INTENT_MAP = WaymoE2EDataset.INTENT_MAP

#     # Model setup
#     B, V, C, H, W = 8, 8, 3, 224, 224
#     M, NM, D = 16, 128, 1024
#     bev_dim, bev_h, bev_w = 64, 16, 16

#     model = MultiViewQFormer(
#         num_views=V, num_queries_per_view=M,
#         vision_model_name="openai/clip-vit-large-patch14",
#         num_layers=2, num_heads=16
#     ).to(device)

#     encoder = BEVFeatureEncoder(
#         num_views=V, num_queries_per_view=M,
#         query_dim=D, bev_dim=bev_dim,
#         bev_h=bev_h, bev_w=bev_w
#     ).to(device)

#     tf_module = TemporalFusion(num_positions=NM, d_model=D).to(device)
#     pose_encoder = PoseTokenEncoder(input_dim=64, d_model=D).to(device)
#     route_encoder = RouteTokenEncoder(input_dim=16, d_model=D).to(device)
#     planner = PlannerHead4D(model_name="google/flan-t5-large", d_model=D).to(device)
#     DPT = DepthEstimator()

#     # Processing one batch
#     for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(1):
#         start_time = time.time()

#         images = torch.from_numpy(images.numpy()).to(device)
#         pose_token = torch.from_numpy(pose_token.numpy()).to(device)
#         routing_token = torch.from_numpy(routing_token.numpy()).to(device)

#         print("Stacked Image shape:", images.shape)
#         # print("Intent:", intent[0].item(), "->", INTENT_MAP.get(int(intent[0]), "Unknown"))
#         # print("Past states shape:", past_states[0].shape)
#         # print("Future states shape:", future_states[0].shape)
#         # print("Pose token shape:", pose_token.shape)
#         # print("Routing token shape:", routing_token.shape)

#         # Estimate depth for each image in batch
#         # depth_maps = [DPT.estimate_depth_batch(images[b]) for b in range(B)]
#         # depth_maps = torch.stack(depth_maps).unsqueeze(-1).repeat(1, 1, 1, 1, 3)
#         # depth_maps = depth_maps.permute(0, 1, 4, 2, 3).to(device)

#         # Estimate depth for each image in batch
#         dimages = images.permute(0, 1, 4, 2, 3)
#         with torch.no_grad():
#             depth_maps = DPT.estimate_feature_batch(dimages)  # (B, V, 224, 224)
#             depth_maps = depth_maps.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (B, V, 3, 224, 224)

#         depth_maps = depth_maps.to(device)
#         print(depth_maps.shape)
#         # Depth-based visual encoding
#         depth_tokens = model(depth_maps)
#         depth_emb = encoder(depth_tokens)
#         print("Depth Tokens:", depth_tokens.shape)
#         print("Depth Embeddings:", depth_emb.shape)

#         # Normal visual encoding
#         images = images.permute(0, 1, 4, 2, 3).to(device)
#         visual_tokens = model(images)
#         bev_emb = encoder(visual_tokens)
#         time_embedded = tf_module(visual_tokens)

#         print("Visual Tokens:", visual_tokens.shape)
#         print("BEV Embeddings:", bev_emb.shape)
#         print("Temporal Tokens:", time_embedded.shape)

#         pose_emb = pose_encoder(pose_token)
#         route_emb = route_encoder(routing_token)

#         # Planning
#         waypoints_mean, waypoints_var = planner(bev_emb, time_embedded, pose_emb, route_emb, depth_emb)
#         print("Waypoints mean shape:", waypoints_mean.shape)

#         end_time = time.time()
#         print(f"Inference time: {end_time - start_time:.4f} seconds")

import os
import sys
import time
import warnings
import numpy as np
import torch

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
    for images, intent, past_states, future_states, pose_token, routing_token in train_ds:
        start_time = time.time()

        # Move tensors to device
        images = torch.from_numpy(images.numpy()).to(device)
        pose_token = torch.from_numpy(pose_token.numpy()).to(device)
        future_states = torch.from_numpy(future_states.numpy()).to(device)
        routing_token = torch.from_numpy(routing_token.numpy()).to(device)

        #print("Stacked Image shape:", images.shape)

        # --- Depth Estimation ---
        # Permute images for depth estimator: (B, V, H, W, C) -> (B, V, C, H, W)
        dimages = images.permute(0, 1, 4, 2, 3)
        with torch.no_grad():
            all_depths = []
            for i in range(dimages.shape[0]):
                one_sample = dimages[i : i+1]           # shape (1, 8, 224, 224, 3)
                depths_i = DPT.estimate_depth_batch(one_sample)  # (1, 8, 224, 224)
                all_depths.append(depths_i)
            depth_maps = torch.cat(all_depths, dim=0)  # (8, 8, 224, 224)
            depth_maps = depth_maps.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (B, V, 3, 224, 224)
        depth_maps = depth_maps.to(device)
        #print("Depth maps shape:", depth_maps.shape)

        # --- Depth-based Visual Encoding ---
        depth_tokens = model(depth_maps)
        depth_emb = encoder(depth_tokens)
        #print("Depth Tokens:", depth_tokens.shape)
        #print("Depth Embeddings:", depth_emb.shape)

        # --- Normal Visual Encoding ---
        images = images.permute(0, 1, 4, 2, 3).to(device)  # (B, V, C, H, W)
        visual_tokens = model(images)
        bev_emb = encoder(visual_tokens)
        time_embedded = tf_module(visual_tokens)
        #print("Visual Tokens:", visual_tokens.shape)
        #print("BEV Embeddings:", bev_emb.shape)
       #print("Temporal Tokens:", time_embedded.shape)

        # --- Pose & Route Encoding ---
        pose_emb = pose_encoder(pose_token)
        route_emb = route_encoder(routing_token)

        # --- Planning ---
        # waypoints_mean, waypoints_var = planner(
        #     bev_emb, time_embedded, pose_emb, route_emb, depth_emb
        # )
        # print("Waypoints mean shape:", waypoints_mean.shape)

        # end_time = time.time()
        # print(f"Inference time: {end_time - start_time:.4f} seconds")

        # print("Future States array:", future_states[0])
        # print("Predicted Future States array:", waypoints_mean[0])
        means, vars_ = [], []
        B_actual = bev_emb.shape[0]
        for i in range(B_actual):
            bev_i   = bev_emb[i : i+1]
            temp_i  = time_embedded[i : i+1]
            pose_i  = pose_emb[i : i+1]
            route_i = route_emb[i : i+1]
            depth_i = depth_emb[i : i+1]

            torch.cuda.empty_cache()
            with torch.no_grad():
                mean_i, var_i = planner(bev_i, temp_i, pose_i, route_i, depth_i)

            means.append(mean_i)   # (1, N, 2)
            vars_.append(var_i)

        # Re-stack into full batch
        waypoints_mean = torch.cat(means, dim=0)  # (B, N, 2)
        waypoints_var  = torch.cat(vars_,  dim=0)  # (B, N, 2)
        #print("Waypoints mean shape:", waypoints_mean.shape)

        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f} seconds")

        #print("Future States array:", future_states[0])
        #print("Predicted Future States array:", waypoints_mean[0])
        #print(waypoints_var[0])