import os
import warnings

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from waymo_e2e.data import WaymoDatasetLoader, WaymoE2EDataset
from waymo_e2e.losses import trajectory_loss
from waymo_e2e.architectures.baseline import E2EConfig, E2EVisualPlanner

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPOCHS = 20
STEPS_PER_EPOCH = 200
BATCH_SIZE = 8
NUM_TEMPORAL_FRAMES = 5  # Phase 1: short history window (must match dataset + model)
LR = 3e-4
GRAD_CLIP = 1.0
SAVE_EVERY = 5  # epochs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Torch using: {device}  |  GPUs available: {num_gpus}")

    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    dataset_builder = WaymoE2EDataset(
        batch_size=BATCH_SIZE,
        num_temporal_frames=NUM_TEMPORAL_FRAMES,
    )
    train_ds = dataset_builder.build_dataset(train_files[:4])  # use first 4 shards

    cfg = E2EConfig(
        num_views=3,
        num_queries_per_view=16,
        d_model=1024,
        bev_dim=64,
        bev_h=32,
        bev_w=32,
        num_temporal_frames=NUM_TEMPORAL_FRAMES,
        vision_backbone="clip",
        vision_model_name="openai/clip-vit-base-patch16",
        freeze_backbone=True,
        unfreeze_last_n_layers=4,   # unfreeze last 4 transformer blocks
        use_thin_depth_fusion=True,
        planner_type="light",
        planner_tf_layers=2,
        planner_tf_heads=16,
        use_legacy_dpt_bev_branch=False,
    )
    model = E2EVisualPlanner(cfg)

    # Multi-GPU: wrap with DataParallel when >1 GPU available
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Separate LR for backbone vs the rest (backbone gets 10× lower LR)
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": LR / 10},
            {"params": other_params,    "lr": LR},
        ],
        weight_decay=1e-4,
    )

    total_steps = EPOCHS * STEPS_PER_EPOCH
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[LR / 10, LR],
        total_steps=total_steps,
        pct_start=0.05,           # 5% warmup
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        steps_this_epoch = 0

        for images, _intent, past_states, future_states, pose_token, routing_token in train_ds.take(STEPS_PER_EPOCH):
            images = torch.from_numpy(images.numpy()).to(device)
            past_states = torch.from_numpy(past_states.numpy()).to(device)
            pose_token = torch.from_numpy(pose_token.numpy()).to(device)
            future_states = torch.from_numpy(future_states.numpy()).to(device)
            routing_token = torch.from_numpy(routing_token.numpy()).to(device)
            if images.dim() == 6:
                # (B, T, V, H, W, C) → (B, T, V, C, H, W)
                images = images.permute(0, 1, 2, 5, 3, 4)
            else:
                images = images.permute(0, 1, 4, 2, 3)  # (B, V, H, W, C) → (B, V, C, H, W)

            optimizer.zero_grad()
            waypoints_mean, _ = model(images, pose_token, routing_token)
            loss = trajectory_loss(waypoints_mean, future_states, past_xy=past_states[..., :2])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            steps_this_epoch += 1
            global_step += 1

        avg_loss = total_loss / max(steps_this_epoch, 1)
        current_lr = scheduler.get_last_lr()[-1]
        print(f"Epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}  lr={current_lr:.2e}  steps={steps_this_epoch}")

        if (epoch + 1) % SAVE_EVERY == 0:
            state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            path = f"e2e_planner_epoch_{epoch + 1}.pt"
            torch.save(state, path)
            print(f"  Saved checkpoint → {path}")

    print("Training completed.")


if __name__ == "__main__":
    main()
