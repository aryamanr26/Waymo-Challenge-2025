import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.stderr = open(os.devnull, 'w') 
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import time
import numpy as np
# Ensure PyTorch uses GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch using:", device)
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset
from vision_encoder import MultiViewQFormer
from bev import BEVFeatureEncoder
from tfusion import TemporalFusion
# from depth_model import DepthPredictor
from embedding import PoseTokenEncoder, RouteTokenEncoder
from planner_head import PlannerHead3D, PlannerHead4D
from depth_est import DepthEstimator
# tf.config.experimental.set_visible_devices([], 'GPU')  # Disable GPU for TF

if __name__ == "__main__":
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Print device", device)

    INTENT_MAP = WaymoE2EDataset.INTENT_MAP

    dataset_builder = WaymoE2EDataset(batch_size=8)
    train_ds = dataset_builder.build_dataset(train_files[0])
    # ------------------------------
    # Neatly organized model setup
    # ------------------------------

    # Batch and camera/view configuration
    B, V, C, H, W = 8, 8, 3, 224, 224  # batch, views, channels, height, width
    M = 16                             # queries per view
    NM, D = 128, 1024                  # temporal tokens, hidden dimension
    bev_dim, bev_h, bev_w = 64, 16, 16

    # ------------------------------
    # Multi-view query former
    # ------------------------------
    model = MultiViewQFormer(num_views=V,
        num_queries_per_view=M,
        vision_model_name="openai/clip-vit-large-patch14",
        num_layers=2,
        num_heads=16  # CLIP ViT-L uses 16 heads
    ).to(device)

    # ------------------------------
    # BEV feature encoder
    # ------------------------------
    encoder = BEVFeatureEncoder(
        num_views=V,
        num_queries_per_view=M,
        query_dim=D,
        bev_dim=bev_dim,
        bev_h=bev_h,
        bev_w=bev_w,
    ).to(device)

    # ------------------------------
    # Temporal fusion module
    # ------------------------------
    tf_module = TemporalFusion(
        num_positions=NM,
        d_model=D
    ).to(device)

    # ------------------------------
    # Token encoders
    # ------------------------------
    pose_encoder = PoseTokenEncoder(input_dim=64, d_model=D).to(device)
    route_encoder = RouteTokenEncoder(input_dim=16, d_model=D).to(device)

    # ------------------------------
    # Depth prediction and planning
    # ------------------------------
    # predictor = DepthPredictor()
    planner = PlannerHead4D(
        model_name="google/flan-t5-large",
        d_model=D
    ).to(device)

    DPT = DepthEstimator()
    
    # model = torch.jit.trace(model)
    # encoder = torch.jit.trace(encoder)
    # tf_module = torch.jit.trace(tf_module)
    # planner = torch.jit.trace(planner)
    
    # ------------------------------
    # Initialize step counter
    # ------------------------------
    depth_maps = []
    i = 0


    for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(1):
        start_time = time.time()
        images = torch.from_numpy(images.numpy()).to(device)
        # intent = torch.from_numpy(intent.numpy())
        # past_states = torch.from_numpy(past_states.numpy())
        # future_states = torch.from_numpy(future_states.numpy())
        pose_token = torch.from_numpy(pose_token.numpy()).to(device)
        routing_token = torch.from_numpy(routing_token.numpy()).to(device)

        print("Stacked Image shape:", images.shape)
        print("Intent:", intent[0].item(), "->", INTENT_MAP.get(int(intent[0]), "Unknown"))
        print("Past states shape:", past_states[0].shape)
        print("Future states shape:", future_states[0].shape)
        print("Pose token shape:", pose_token.shape)
        print("Routing token shape:", routing_token.shape)

        for num in range(8):
            depth_map = DPT.estimate_depth_batch(images[num])
            depth_maps.append(depth_map)
            # print(f"Depth map for image {num}:", depth_map.shape)
            time.sleep(0.1)

        depth_maps = torch.stack(depth_maps, dim = 0) ## 8, 8, 224, 224
        # print(depth_maps.shape)
        depth_maps = depth_maps.unsqueeze(-1).repeat(1, 1, 1, 1, 3) ## 8,8,224,224,3
        depth_maps = depth_maps.permute(0, 1, 4, 2, 3).to(device) ## 8, 8, 3, 224, 224

        depth_tokens = model(depth_maps).to(device)
        print("Final Depth Tokens: ", depth_tokens.shape)

        depth_emb = encoder(depth_tokens).to(device)
        print("Final Depth Embeddings: ", depth_emb.shape)

        modified_image = images.permute(0, 1, 4, 2, 3)
        modified_image = modified_image.to(device)
        # print(modified_image.shape)

        visual_tokens = model(modified_image).to(device)  # (B, V*M, d_model)
        print("Final Visual Tokens: ", visual_tokens.shape)  # e.g., (4, 8*16, 1024) for ViT-L/14

        bev_emb = encoder(visual_tokens).to(device)
        print("Final BEV tokens:", bev_emb.shape)
    
        time_embedded = tf_module(visual_tokens).to(device)
        print("Final Temporal tokens:", time_embedded.shape)

        img_tensor = modified_image[0]
        img_tensor = img_tensor.to(device)

        print("Image Tensor Shape: ", img_tensor.shape)
        
        # depth_map = predictor.predict_depth(img_tensor)
        # depth_map = depth_map.to(device)
        # print("Predicted depth shape:", depth_map.shape)

        # depth_maps = [DPT.estimate_depth_batch(img) for img in images]
        # depth_maps = np.stack(depth_maps)

        # one_image = images[0]
        # depth_map = DPT.estimate_depth_batch(one_image)
        # depth_map = np.stack(depth_map)  # Convert list of arrays to a single NumPy array
        # depth_map = torch.from_numpy(depth_map).to(device)

        # print("Predicted DPT shape:", depth_map.shape)

        # Optional: compute loss between two depths (demo)
        # dummy_loss = predictor.photometric_loss(depth_map, depth_map * 0.95)
        # print(f"SSIM-based photometric loss: {dummy_loss.item():.7f}")

        pose_emb = pose_encoder(pose_token).to(device)
        route_emb = route_encoder(routing_token).to(device)
       
        waypoints_mean, waypoints_var = planner(bev_emb, time_embedded, pose_emb, route_emb, depth_emb)
        i += 1

        print("Waypoints for {} image -> mean:".format(i), waypoints_mean.shape)
        # print("Waypoints for {} image -> variance:".format(i), waypoints_var.shape)
        end_time = time.time()
        print(f"Iteration {i} took {end_time - start_time:.4f} seconds")