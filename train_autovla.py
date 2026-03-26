"""
AutoVLA SFT training: Qwen2.5-VL-7B (LoRA) + regression head.

Launch command (4 GPUs):
    accelerate launch --num_processes 4 train_autovla.py

Or single GPU:
    python train_autovla.py

Phase: Supervised Fine-Tuning (SFT) on Waymo WOD-E2E dataset.
Next:  GRPO RL with RFS-proxy reward  →  train_autovla_rl.py
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from waymo_e2e.data import WaymoDatasetLoader, WaymoE2EDataset
from waymo_e2e.losses import trajectory_loss
from waymo_e2e.architectures.autovla import QwenE2EPlanner, QwenPlannerConfig

# ---------------------------------------------------------------------------
# Hyper-parameters — edit here or override with a config file
# ---------------------------------------------------------------------------
MODEL_NAME         = "Qwen/Qwen2.5-VL-7B-Instruct"   # or "3B" for faster iteration
LORA_RANK          = 16
LORA_ALPHA         = 32
LOAD_IN_4BIT       = False   # set True for QLoRA if VRAM is tight

EPOCHS             = 10
STEPS_PER_EPOCH    = 300
BATCH_SIZE         = 2       # per GPU; effective = BATCH_SIZE × num_gpus × GRAD_ACCUM
GRAD_ACCUM         = 4       # accumulate before step (effective batch ≈ 32 on 4 GPUs)
LR                 = 2e-4    # LoRA + regression head
BACKBONE_LR_SCALE  = 0.1     # LoRA params get LR × this vs regression head
GRAD_CLIP          = 1.0
SAVE_EVERY         = 2       # epochs
SEED               = 42

NUM_TRAIN_SHARDS   = 16      # how many TFRecord shards to use for training
# ---------------------------------------------------------------------------


def build_dataset(num_temporal_frames: int = 1):
    loader = WaymoDatasetLoader()
    train_files, val_files, _ = loader.get_file_lists()
    ds_builder = WaymoE2EDataset(
        img_size=(448, 448),           # higher res for Qwen2.5-VL
        batch_size=BATCH_SIZE,
        num_temporal_frames=num_temporal_frames,
    )
    train_ds = ds_builder.build_dataset(train_files[:NUM_TRAIN_SHARDS])
    return train_ds


def main():
    # -----------------------------------------------------------------------
    # Accelerator — handles device placement, DDP, mixed precision
    # -----------------------------------------------------------------------
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir="./logs_autovla",
    )
    set_seed(SEED)
    accelerator.print(
        f"\nAutoVLA SFT | {accelerator.num_processes} GPU(s) | "
        f"effective batch = {BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM}\n"
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    cfg = QwenPlannerConfig(
        model_name=MODEL_NAME,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        load_in_4bit=LOAD_IN_4BIT,
        torch_dtype="bfloat16",
        num_waypoints=20,
        regress_hidden_dim=1024,
        image_size=448,
    )
    with accelerator.main_process_first():
        model = QwenE2EPlanner(cfg)

    # Gradient checkpointing to save activation memory
    model.vlm.gradient_checkpointing_enable()

    # -----------------------------------------------------------------------
    # Optimizer — separate LRs for LoRA params vs regression head
    # -----------------------------------------------------------------------
    lora_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "regress_head" in name:
            head_params.append(p)
        else:
            lora_params.append(p)   # all other trainable params = LoRA adapters

    optimizer = AdamW(
        [
            {"params": lora_params, "lr": LR * BACKBONE_LR_SCALE},
            {"params": head_params, "lr": LR},
        ],
        weight_decay=1e-2,
        betas=(0.9, 0.95),
    )

    total_steps = (EPOCHS * STEPS_PER_EPOCH) // GRAD_ACCUM
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[LR * BACKBONE_LR_SCALE, LR],
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    train_ds = build_dataset()

    # -----------------------------------------------------------------------
    # Prepare with accelerator (handles device placement + DDP wrapping)
    # -----------------------------------------------------------------------
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        steps_done = 0

        for images_tf, intents_tf, past_xy_tf, future_states_tf, _, _ in train_ds.take(STEPS_PER_EPOCH):
            # -- Convert TF tensors to numpy --
            images_np    = images_tf.numpy()        # (B, V, H, W, 3) float32 [0,1]
            intents_np   = intents_tf.numpy()       # (B,) int32
            past_xy_np   = past_xy_tf.numpy()       # (B, 64, 2) float32
            future_np    = future_states_tf.numpy() # (B, T, 3) float32

            future_torch = torch.from_numpy(future_np).to(accelerator.device)

            # -- Build VLM inputs on the fly (image→PIL + processor) --
            # Use the underlying module when wrapped by accelerator/DDP
            raw_model = accelerator.unwrap_model(model)
            inputs = raw_model.prepare_inputs(
                images_bvhwc=images_np,
                intents=intents_np,
                past_xy=past_xy_np,
                device=accelerator.device,
            )

            with accelerator.accumulate(model):
                waypoints = model(inputs)           # (B, 20, 3)
                loss = trajectory_loss(
                    waypoints,
                    future_torch,
                    past_xy=torch.from_numpy(past_xy_np).to(accelerator.device),
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            total_loss += loss.item()
            steps_done += 1

        avg_loss = total_loss / max(steps_done, 1)
        current_lr = scheduler.get_last_lr()[-1]
        accelerator.print(
            f"Epoch {epoch + 1}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  lr={current_lr:.2e}  global_step={global_step}"
        )

        # -- Checkpoint: save LoRA adapters + regression head only --
        if accelerator.is_main_process and (epoch + 1) % SAVE_EVERY == 0:
            save_dir = f"checkpoints_autovla/epoch_{epoch + 1}"
            os.makedirs(save_dir, exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)
            # Save LoRA adapters (tiny — only trained delta weights)
            unwrapped.vlm.save_pretrained(os.path.join(save_dir, "lora_adapters"))
            # Save regression head
            torch.save(
                unwrapped.regress_head.state_dict(),
                os.path.join(save_dir, "regress_head.pt"),
            )
            accelerator.print(f"  Saved checkpoint → {save_dir}/")

    accelerator.print("SFT training complete.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
