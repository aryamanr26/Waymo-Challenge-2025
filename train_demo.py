import os
import torch
from torch import optim
# Ensure PyTorch uses GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch using:", device)
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset
from vision_encoder import MultiViewQFormer
from bev import BEVFeatureEncoder
from tfusion import TemporalFusion
from depth.depth_model import DepthPredictor
from embedding import PoseTokenEncoder, RouteTokenEncoder
from planner_head import PlannerHead3D
# tf.config.experimental.set_visible_devices([], 'GPU')  # Disable GPU for TF

if __name__ == "__main__":
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Print device", device)

    INTENT_MAP = WaymoE2EDataset.INTENT_MAP

    dataset_builder = WaymoE2EDataset(batch_size=4)
    train_ds = dataset_builder.build_dataset(train_files[0])
    # ------------------------------
    # Neatly organized model setup
    # ------------------------------

    # Batch and camera/view configuration
    B, V, C, H, W = 4, 8, 3, 224, 224  # batch, views, channels, height, width
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
    pose_encoder = PoseTokenEncoder(input_dim=64, d_model=D)
    route_encoder = RouteTokenEncoder(input_dim=16, d_model=D)

    # ------------------------------
    # Depth prediction and planning
    # ------------------------------
    predictor = DepthPredictor()
    planner = PlannerHead3D(
        model_name="google/flan-t5-large",
        d_model=D
    ).to(device)

    # ------------------------------
    # Initialize step counter
    # ------------------------------
    i = 0

    encoder.to(device)
    tf_module.to(device)
    pose_encoder.to(device)
    route_encoder.to(device)
    predictor.to(device)
    planner.to(device)

    # ----------------------
    # Optimizer setup
    # ----------------------
    params = list(encoder.parameters()) + \
            list(tf_module.parameters()) + \
            list(pose_encoder.parameters()) + \
            list(route_encoder.parameters()) + \
            list(predictor.parameters()) + \
            list(planner.parameters())

    optimizer = optim.Adam(params, lr=1e-4)
    num_epochs = 8

    # ----------------------
    # Training Loop
    # ----------------------
    for epoch in range(num_epochs):
        for batch in train_ds:
            # Preprocessing: move tensors to device
            images, intent, past_states, future_states, pose_token, routing_token = [
                torch.from_numpy(x.numpy()).to(device) for x in batch
            ]

            # Forward pass
            modified_image = images.permute(0, 1, 4, 2, 3)  # (B, V, C, H, W)
            visual_tokens = model(modified_image)

            bev_emb = encoder(visual_tokens)
            time_embedded = tf_module(visual_tokens)

            img_tensor = modified_image[0]  # Choose one for depth
            depth_map = predictor.predict_depth(img_tensor)

            pose_emb = pose_encoder(pose_token)
            route_emb = route_encoder(routing_token)

            waypoints_mean, waypoints_var = planner(
                bev_emb, time_embedded, pose_emb, route_emb
            )

            # ----------------------
            # Compute custom loss
            # ----------------------
            # TODO: Replace with your custom loss
            loss = torch.tensor(0.0, device=device)

            # ----------------------
            # Backward + Optimize
            # ----------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
