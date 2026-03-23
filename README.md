# Waymo Challenge 2025 — End-to-End Visual Planning

Experiments for **end-to-end trajectory prediction** from multi-camera images (Waymo Open Dataset E2E camera split). The stack is designed to stay **trainable**: one vision backbone, optional lightweight depth fusion, and a **small Transformer + MLP** planner instead of a large language model.

## Architecture (current defaults)

1. **Vision** — Single forward per frame batch: **CLIP**, **DINOv2**, or **SigLIP** (`transformers`), with `[0, 1]` inputs normalized per backbone.
2. **Projection** — Patch embeddings are projected to `d_model` (default **1024**) when the backbone width differs (e.g. CLIP-B → 1024 for the rest of the stack).
3. **Thin depth fusion** — A small MLP adds **depth-style residuals to patch tokens** (late fusion). No second ViT pass and no DPT in the default path.
4. **Multi-view Q-Former** — Learned queries cross-attend to all views’ patch tokens; per-camera embeddings are added.
5. **BEV encoder** — Per-view tokens are mapped to a coarse BEV grid, warped with learnable homographies, summed, and projected.
6. **Temporal bias** — Learned additive embeddings per (camera × query) slot (not multi-frame video yet).
7. **Pose / route** — MLP encodings from past-state token and intent embedding.
8. **Planner** — **`LightTrajectoryHead`** (default): small `TransformerEncoder` over fused tokens → attention pooling → MLP → `num_waypoints × (x, y, z)`. Optional legacy **`PlannerHead4D`** (Flan-T5 encoder) via config.

Legacy (off by default): **DPT depth maps** fed through a **second** full vision + BEV pass into an extra planner branch (`use_legacy_dpt_bev_branch=True`).

## Repository layout

| Path | Purpose |
|------|---------|
| `waymo_e2e/` | Installable package |
| `waymo_e2e/data/` | GCS file listing (`WaymoDatasetLoader`), TF `tf.data` E2E pipeline (`WaymoE2EDataset`) |
| `waymo_e2e/models/` | Vision Q-Former, BEV, temporal fusion, embeddings, depth helper, planner heads |
| `waymo_e2e/losses/` | `ade_fde_loss` |
| `waymo_e2e/pipeline.py` | `E2EConfig`, `E2EVisualPlanner` |
| Root shims (`vision_encoder.py`, `bev.py`, …) | Re-export from `waymo_e2e` for older import paths |
| `waymo-open-dataset/` | Expected under repo root for protobuf / ops (see `waymo_e2e/data/gcs_loader.py`) |

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
- **opencv-python**, **numpy**

## Data access

`WaymoDatasetLoader` lists TFRecords from the **Waymo Open Dataset** E2E camera bucket (see `waymo_e2e/data/gcs_loader.py`). You need **Google Cloud** credentials configured if reading from GCS.

`WaymoE2EDataset` parses **E2EDFrame** records: resized camera images, past states, future xyz trajectory, intent, and derived pose/route tokens. **`num_views` in config should match the dataset** (the bundled parser stacks **3** front/side images by default).

## Usage

### Full model from config

```python
from waymo_e2e.pipeline import E2EConfig, E2EVisualPlanner

cfg = E2EConfig(
    num_views=3,
    d_model=1024,
    vision_backbone="clip",
    vision_model_name="openai/clip-vit-base-patch16",
    use_thin_depth_fusion=True,
    planner_type="light",
    use_legacy_dpt_bev_branch=False,
)
model = E2EVisualPlanner(cfg).cuda()

# images: (B, V, 3, H, W) float in [0, 1]
waypoints, _ = model(images_bvchw, pose_token, routing_token)  # (B, T, 3)
```

### Training script

```bash
python train.py
```

Uses `E2EVisualPlanner`, `ade_fde_loss`, and a short `tf.data` loop (adjust epochs / steps as needed).

### One-batch inference / smoke test

```bash
python main.py
```

### Other entry points

- `train_demo.py` — alternate demo loop with components wired manually.
- `motion/qwen.py` — standalone Qwen2.5-VL smoke test (not part of the default training stack).

### Key `E2EConfig` fields

| Field | Meaning |
|-------|---------|
| `num_views` | Must match stacked cameras in your dataset (default **3**). |
| `vision_backbone` | `"clip"` \| `"dinov2"` \| `"siglip"`. |
| `vision_model_name` | Hugging Face model id (e.g. `facebook/dinov2-small`, `google/siglip-base-patch16-224`). |
| `freeze_backbone` | Freeze vision weights (default `True`). |
| `use_thin_depth_fusion` | MLP fusion on patch tokens (default `True`). |
| `planner_type` | `"light"` (default) or `"t5"` (legacy Flan-T5 encoder head). |
| `planner_tf_layers` / `planner_tf_heads` / `planner_hidden_dim` | Light planner depth and width. |
| `use_legacy_dpt_bev_branch` | If `True`, runs frozen DPT + second ViT+BEV branch (heavy). |

### Imports (package vs shims)

Prefer:

```python
from waymo_e2e.pipeline import E2EVisualPlanner, E2EConfig
from waymo_e2e.losses import ade_fde_loss
```

Root-level modules such as `from vision_encoder import MultiViewQFormer` still work as thin re-exports.

## Evaluation target

The intended supervision is **future trajectory** over the challenge horizon (e.g. **5 s**), with losses such as **ADE / FDE** on predicted waypoints vs. labels (`waymo_e2e/losses/trajectory.py`). Align shapes and horizon with the official **Waymo End-to-End Driving Challenge** submission format when exporting results.

## License / attribution

Waymo Open Dataset is subject to its own terms. Model checkpoints (CLIP, DINOv2, SigLIP, optional T5/DPT) follow their respective licenses.
