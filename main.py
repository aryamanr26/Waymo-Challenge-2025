import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow
import torch
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset
from vision_encoder import MultiViewQFormer
from bev import BEVFeatureEncoder
from tfusion import TemporalFusion
from depth_model import DepthPredictor
# tf.config.experimental.set_visible_devices([], 'GPU')  # Disable GPU for TF

if __name__ == "__main__":
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Print device", device)

    INTENT_MAP = WaymoE2EDataset.INTENT_MAP

    dataset_builder = WaymoE2EDataset(batch_size=4)
    train_ds = dataset_builder.build_dataset(train_files[:2])

    B, V, C, H, W = 4, 8, 3, 224, 224
    M = 16  # queries per view
    

    for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(1):
        images = torch.from_numpy(images.numpy())
        intent = torch.from_numpy(intent.numpy())
        past_states = torch.from_numpy(past_states.numpy())
        future_states = torch.from_numpy(future_states.numpy())
        pose_token = torch.from_numpy(pose_token.numpy())
        routing_token = torch.from_numpy(routing_token.numpy())

        print("Stacked Image shape:", images.shape)
        print("Intent:", intent[0].item(), "->", INTENT_MAP.get(int(intent[0]), "Unknown"))
        print("Past states shape:", past_states[0].shape)
        print("Future states shape:", future_states[0].shape)
        print("Pose token shape:", pose_token.shape)
        print("Routing token shape:", routing_token.shape)
    
    modified_image = images.permute(0, 1, 4, 2, 3)
    print(modified_image.shape)
    model = MultiViewQFormer(
         num_views=V,
         num_queries_per_view=M,
         vision_model_name="openai/clip-vit-large-patch14",
         num_layers=2,
         num_heads=16,  # match ViT-L’s 16 attention heads
     ).to(device)
    visual_tokens = model(modified_image)  # (B, V*M, d_model)
    print(visual_tokens.shape)  # e.g., (4, 8*16, 1024) for ViT-L/14

    B, V, C, H, W = 4, 8, 3, 224, 224
    bev_dim, bev_h, bev_w = 64, 16, 16
    M, D = 16, 1024
    encoder = BEVFeatureEncoder(
        num_views=V,
        num_queries_per_view=M,
        query_dim=D,
        bev_dim=bev_dim,
        bev_h=bev_h,
        bev_w=bev_w,
    )
    bev_tokens = encoder(visual_tokens)
    print("Final BEV tokens:", bev_tokens.shape)

    B, NM, D_t = 4, 128, 1024
    tf_module = TemporalFusion(num_positions=NM, d_model=D_t)
    time_embedded = tf_module(visual_tokens)
    print("Final Temporal tokens:", time_embedded.shape)

    predictor = DepthPredictor()
    img_tensor = modified_image[0]
    print("Image Tensor Shape: ", img_tensor.shape)
    depth_map = predictor.predict_depth(img_tensor)
    print("Predicted depth shape:", depth_map.shape)

    # Optional: compute loss between two depths (demo)
    dummy_loss = predictor.photometric_loss(depth_map, depth_map * 0.95)
    print(f"SSIM-based photometric loss: {dummy_loss.item():.7f}")