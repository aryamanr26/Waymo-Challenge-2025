# Waymo Challenge 2025 — End-to-End Visual Planning

Experiments for **end-to-end trajectory prediction** from multi-camera images (Waymo Open Dataset E2E camera split). The stack is designed to stay **trainable**: one vision backbone, optional lightweight depth fusion, and a **small Transformer + MLP** planner instead of a large language model.

There are **two training tracks**:

| Track | Script | Model |
|-------|--------|--------|
| **Phase 1 (CNN + Q-Former + BEV)** | `train.py` | `E2EVisualPlanner` (CLIP/DINOv2 + light planner) |
| **AutoVLA-style SFT** | `train_autovla.py` | **Qwen2.5-VL** + LoRA + waypoint regression head (`QwenE2EPlanner`) |

See `claude.md` for roadmap (DiffusionDrive head, GRPO / RFS-proxy RL, etc.).

## What’s implemented (Phase 1 foundation)

Recent work focused on **Phase 1** from the project plan (`claude.md`):

- **Partial backbone training** — Last **4** transformer blocks of the vision backbone (CLIP/ViT by default) are unfrozen; backbone parameters use **10× lower** learning rate than the rest of the model.
- **Finer BEV** — Default grid is **32×32** (was 16×16) for richer spatial resolution before the planner.
- **Short temporal context** — **5 frames** of history are stacked from consecutive TFRecords (`tf.data` windows). The **Q-Former** runs per-frame; the **TemporalCrossAttention** module fuses history so the **current** timestep (last frame) **queries** all past frame tokens.
- **Trajectory loss** — Beyond ADE/FDE: **progressive temporal weighting**, **heading consistency** (XY), **velocity smoothness** (jerk), and a **collision-proxy** term that penalizes predicted waypoints passing too close to the **ego past polyline** in XY (padded to a fixed shape for batching).
- **Training loop** — `AdamW`, **OneCycleLR** (warmup + cosine), **gradient clipping**, **DataParallel** when multiple GPUs are present, periodic checkpoints.

Optional **single-frame** mode still works: set `num_temporal_frames=1` in `WaymoE2EDataset` and pass images as shape `(B, V, 3, H, W)`.

## Pipeline (data → waypoints)

End-to-end flow is implemented in **`E2EVisualPlanner`** (`waymo_e2e/pipeline.py`):

```text
Multi-camera images [+ optional T history frames]
    → MultiViewQFormer
        CLIP/DINOv2/SigLIP patch tokens → project to d_model
        → optional thin depth MLP on patches
        → per-camera embeddings
        → TransformerDecoder: learned queries attend to all views’ patches
        → per-frame Q-Former tokens (B, N, D) or (B, T, N, D)
    → TemporalCrossAttention
        fuses (B, T, N, D) into (B, N, D) using last-frame queries over full history
        (if T=1, equivalent to + learned per-slot bias)
    → BEVFeatureEncoder
        maps fused tokens to a BEV grid (learnable homographies), sum over views, project
    → PoseTokenEncoder(past-state token) + RouteTokenEncoder(intent/route token)
    → LightTrajectoryHead (default)
        TransformerEncoder over [BEV tokens ∥ temporal tokens ∥ pose ∥ route]
        → attention pooling → MLP → (num_waypoints × 3) for (x, y, z)
```

**Optional legacy path** (`use_legacy_dpt_bev_branch=True`): frozen DPT depth → second vision + BEV branch into the planner (heavy; off by default).

**Loss** (`waymo_e2e/losses/trajectory.py`): `trajectory_loss(predictions, targets, past_xy=...)` combines weighted ADE/FDE, heading, smoothness, and optional past-path proximity.

**Implementation note:** `BEVFeatureEncoder` expects token shape `(B, N, D)`. When using **multi-frame** input, the Q-Former returns `(B, T, N, D)`. The pipeline therefore runs **`TemporalCrossAttention` first** (collapsing time → `(B, N, D)`), then **`BEVFeatureEncoder`** on the fused tokens. The legacy depth branch keeps its own vision pass separate.

## AutoVLA-style training (Qwen2.5-VL + LoRA)

For **supervised fine-tuning (SFT)** with a **vision-language model** and a small **waypoint regression head** (AutoVLA-style “fast path”):

| Path | Role |
|------|------|
| `waymo_e2e/prompts/waymo_prompt.py` | System/user prompts, intent → language, speed from past XY, waypoint ↔ text helpers for SFT / CoT |
| `waymo_e2e/models/qwen_planner.py` | `QwenPlannerConfig`, `QwenE2EPlanner` — Qwen2.5-VL + PEFT LoRA + MLP head → `(B, 20, 3)` waypoints |
| `train_autovla.py` | **Hugging Face Accelerate** (bf16, gradient accumulation, DDP-friendly for LoRA), differential LR (LoRA ×0.1 vs head), saves **LoRA + head** only every few epochs |

**Run (after `pip install` includes `accelerate`, `peft`, and optional `bitsandbytes` for 4-bit):**

```bash
# One-time Accelerate setup (interactive)
accelerate config

# 4 GPUs
accelerate launch --num_processes 4 train_autovla.py

# Single GPU smoke test
python train_autovla.py
```

Logs default to `./logs_autovla` (TensorBoard). Tune `MODEL_NAME`, `BATCH_SIZE`, `LOAD_IN_4BIT`, and shard count at the top of `train_autovla.py`. **Next step** on the roadmap: RL (e.g. GRPO with RFS-proxy) in a separate script such as `train_autovla_rl.py` (not yet in repo unless added).

## Repository layout

| Path | Purpose |
|------|---------|
| `waymo_e2e/` | Installable package |
| `waymo_e2e/architectures/baseline/` | Baseline namespace (`E2EConfig`, `E2EVisualPlanner`) |
| `waymo_e2e/architectures/autovla/` | AutoVLA namespace (`QwenE2EPlanner`, prompt helpers) |
| `waymo_e2e/data/` | GCS file listing (`WaymoDatasetLoader`), TF `tf.data` E2E pipeline (`WaymoE2EDataset`) |
| `waymo_e2e/models/` | Vision Q-Former, BEV, temporal fusion, embeddings, depth helper, planner heads, **`qwen_planner.py`** (Qwen2.5-VL + LoRA) |
| `waymo_e2e/prompts/` | **`waymo_prompt.py`** — prompts and waypoint text helpers for AutoVLA SFT |
| `waymo_e2e/losses/` | `trajectory_loss`, `ade_fde_loss`, `past_path_proximity_loss` |
| `waymo_e2e/pipeline.py` | `E2EConfig`, `E2EVisualPlanner` |
| `waymo-open-dataset/` | Expected under repo root for protobuf / ops (see `waymo_e2e/data/gcs_loader.py`) |
| `train.py` | Main training entry — Phase 1 CNN + Q-Former + BEV |
| `train_autovla.py` | AutoVLA-style SFT — Qwen2.5-VL + LoRA + regression head (Accelerate) |
| `train_entry.py` | Unified launcher: `--track baseline` or `--track autovla` |
| `main.py` | One-batch inference smoke test (`E2EVisualPlanner`) |
| `claude.md` | Architecture notes, Phase 1/2/3 roadmap, AutoVLA file summary |

## Installation

```bash
cd Waymo-Challenge-2025
pip install -e .
```

Python **3.10+** recommended.

### Dependencies

Install according to your environment (not fully pinned in `pyproject.toml`). Typical stack:

- **PyTorch** (CUDA build if using GPU)
- **transformers**, **torch** — vision backbones and optional T5 planner
- **tensorflow** — `tf.data` for TFRecords
- **waymo-open-dataset** — E2E protos and tools (repo path appended for ops)
- **gcsfs** — listing / reading Waymo buckets if using `gs://` paths
- **Pillow** — JPEG decode for windowed multi-frame batches in `WaymoE2EDataset`
- **numpy**
- **accelerate**, **peft** — for `train_autovla.py` (distributed / mixed precision / LoRA)
- **bitsandbytes** (optional) — 4-bit loading in `QwenE2EPlanner` when `LOAD_IN_4BIT=True`

## Data access

`WaymoDatasetLoader` lists TFRecords from the **Waymo Open Dataset** E2E camera bucket (see `waymo_e2e/data/gcs_loader.py`). You need **Google Cloud** credentials configured if reading from GCS.

`WaymoE2EDataset` parses **E2EDFrame** records: resized camera images, past states (fixed-length padding for batching), future xyz trajectory, intent, and derived pose/route tokens. **`num_views` in config should match the dataset** (the bundled parser stacks **3** front/side images by default).

## How to run the code

### Quick start (single launcher)

```bash
# Baseline stack
python train_entry.py --track baseline

# AutoVLA stack
python train_entry.py --track autovla
```

For multi-GPU AutoVLA, still use `accelerate launch --num_processes 4 train_autovla.py`.

### 1. Environment

Use a conda or venv with **PyTorch + CUDA** (if training on GPU), **TensorFlow**, **transformers**, and `pip install -e .` from this repo.

### 2. Waymo data + GCS

- Ensure **`waymo-open-dataset`** is installed and the **`waymo-open-dataset/`** subtree is available as expected by the loader.
- Configure **GCP credentials** so `WaymoDatasetLoader` can list/read `gs://` paths (see `waymo_e2e/data/gcs_loader.py`).

### 3. Training — Phase 1 (`E2EVisualPlanner`)

```bash
cd Waymo-Challenge-2025
python train.py
```

This uses **4 TFRecord shards** (edit `train_files[:4]` in `train.py` to use more), **5-frame** temporal windows, **32×32 BEV**, **CLIP** ViT-B + partial unfreeze, and saves checkpoints every **5** epochs (`e2e_planner_epoch_*.pt`).

If you **run out of memory**, lower `BATCH_SIZE` in `train.py` or set `NUM_TEMPORAL_FRAMES = 1` (and match `WaymoE2EDataset(..., num_temporal_frames=1)`).

### 4. Training — AutoVLA SFT (`QwenE2EPlanner`)

Uses **Accelerate** (not `DataParallel`) for LoRA-friendly multi-GPU training. See **`train_autovla.py`** header and the **AutoVLA-style training** section above.

### 5. One-batch inference smoke test

```bash
python main.py
```

Uses a single `tf.data` batch and prints shapes and timing (expects **single-frame** data and `bev_h=32`, `bev_w=32` in the config there).

### Other entry points

- `train_autovla.py` — Qwen2.5-VL SFT with LoRA + waypoint head (primary VLM training entry).
- `train_demo.py` — alternate demo loop with components wired manually.
- `motion/qwen.py` — standalone Qwen2.5-VL smoke test (lighter than full `train_autovla.py`).

### Full model from config

```python
from waymo_e2e.pipeline import E2EConfig, E2EVisualPlanner

cfg = E2EConfig(
    num_views=3,
    d_model=1024,
    bev_h=32,
    bev_w=32,
    vision_backbone="clip",
    vision_model_name="openai/clip-vit-base-patch16",
    unfreeze_last_n_layers=4,
    use_thin_depth_fusion=True,
    planner_type="light",
    use_legacy_dpt_bev_branch=False,
)
model = E2EVisualPlanner(cfg).cuda()

# Single frame: (B, V, 3, H, W) float in [0, 1]
# Multi-frame: (B, T, V, 3, H, W) — must match dataset window
waypoints, _ = model(images_bvchw, pose_token, routing_token)  # (B, T_waypoints, 3)
```

### Key `E2EConfig` fields

| Field | Meaning |
|-------|---------|
| `num_views` | Must match stacked cameras in your dataset (default **3**). |
| `bev_h` / `bev_w` | BEV grid size (default **32**). |
| `vision_backbone` | `"clip"` \| `"dinov2"` \| `"siglip"`. |
| `vision_model_name` | Hugging Face model id (e.g. `facebook/dinov2-small`, `google/siglip-base-patch16-224`). |
| `freeze_backbone` | Freeze vision weights (default `True`). |
| `unfreeze_last_n_layers` | Unfreeze last **N** ViT blocks (0 = none). |
| `use_thin_depth_fusion` | MLP fusion on patch tokens (default `True`). |
| `temporal_attn_heads` | Heads for `TemporalCrossAttention`. |
| `num_temporal_frames` | Documented parity with dataset (1 vs multi-frame). |
| `planner_type` | `"light"` (default) or `"t5"` (legacy Flan-T5 encoder head). |
| `planner_tf_layers` / `planner_tf_heads` / `planner_hidden_dim` | Light planner depth and width. |
| `use_legacy_dpt_bev_branch` | If `True`, runs frozen DPT + second ViT+BEV branch (heavy). |

### Imports (package vs shims)

Prefer:

```python
from waymo_e2e.architectures.baseline import E2EVisualPlanner, E2EConfig
from waymo_e2e.data import WaymoDatasetLoader, WaymoE2EDataset
from waymo_e2e.losses import trajectory_loss, ade_fde_loss
```

For AutoVLA:

```python
from waymo_e2e.architectures.autovla import QwenE2EPlanner, QwenPlannerConfig
```

For individual model pieces, import from `waymo_e2e.models.*` (e.g. `MultiViewQFormer` from `waymo_e2e.models.vision_qformer`).

## Evaluation target

The intended supervision is **future trajectory** over the challenge horizon (e.g. **5 s**), with losses such as **ADE / FDE** and the auxiliary terms in `trajectory_loss`. Align shapes and horizon with the official **Waymo End-to-End Driving Challenge** submission format when exporting results.

## License / attribution

Waymo Open Dataset is subject to its own terms. Model checkpoints (CLIP, DINOv2, SigLIP, optional T5/DPT) follow their respective licenses.
