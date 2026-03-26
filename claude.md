· Gen
---
Your Current Architecture — Quick Assessment

Your stack: Frozen CLIP/DINOv2/SigLIP → Patch Projection → Q-Former → BEV (16×16) →
LightTrajectoryHead (2-layer Transformer + MLP)

What's good: Modular, trainable, lightweight planner head, correct intuition on Q-Former for
multi-view fusion.

What's limiting it:
- Frozen backbone = no visual representation learning for driving-specific cues
- BEV 16×16 grid is extremely coarse (loses most spatial detail)
- No temporal modeling (just learned positional embeddings, not actual multi-frame)
- ADE/FDE loss only — misaligned with Waymo's actual evaluation metric (RFS)
- No multi-modal reasoning for long-tail/rare scenarios — which is exactly what WOD-E2E tests

---
What the Top Teams Are Doing (Waymo WOD-E2E 2025)

┌──────┬─────────┬───────────────────────────────────────────────────┬──────┐
│ Rank │  Model  │                   Key Technique                   │ RFS  │
├──────┼─────────┼───────────────────────────────────────────────────┼──────┤
│ 1    │ Poutine │ WOD-E2E + CoVLA data + RL optimizing RFS directly │ 7.99 │
├──────┼─────────┼───────────────────────────────────────────────────┼──────┤
│ 2    │ HMVLM   │ Qwen2.5 backbone + chain-of-thought reasoning     │ 7.74 │
├──────┼─────────┼───────────────────────────────────────────────────┼──────┤
│ 3    │ AutoVLA │ Qwen2.5-VL-3B + GRPO RL                           │ 7.56 │
└──────┴─────────┴───────────────────────────────────────────────────┴──────┘

The single biggest differentiator: RL fine-tuning with human-aligned reward (RFS), not ADE/FDE
regression.

---
Strategic Directions — Ranked by Impact/Feasibility on 4× L40S

---
Direction 1 (Recommended): AutoVLA-style VLM E2E Planner

Based on: AutoVLA (arXiv:2506.13757), #3 Waymo WOD-E2E, open-source

Architecture:
Qwen2.5-VL-3B or 7B (LoRA fine-tuned)
  └─ Multi-view images + routing command + ego status → text prompt
  └─ "Fast mode": outputs (x,y,z) waypoints directly as tokens
  └─ "Slow mode": chain-of-thought → scene analysis → intent → trajectory
  └─ GRPO RL fine-tuning against RFS reward after SFT phase

Why this wins:
- Qwen2.5-VL-3B fits on 1 L40S for inference, 2 L40S for LoRA fine-tuning — you have headroom for
larger batch sizes or 7B
- VLMs understand long-tail scenarios (construction zones, unusual intersections) — exactly what
WOD-E2E tests
- GRPO (Group Relative Policy Optimization) lets you optimize RFS directly
- You already have motion/qwen.py as a starting point

Hardware fit on 4× L40S:
- Qwen2.5-VL-7B full fine-tune: ~2× L40S with gradient checkpointing
- Qwen2.5-VL-7B LoRA: 1× L40S, batch across all 4 GPUs with DDP

---
Direction 2: DiffusionDrive Planner Head Drop-in

Based on: DiffusionDrive (arXiv:2411.15139), 45 FPS, CVPR 2025 Highlight

Architecture — augments your existing pipeline:
Your existing: Vision → Q-Former → BEV tokens
  └─ Replace LightTrajectoryHead with:
     DiffusionDrive head: pre-computed trajectory anchors (multi-mode)
     → 2-step truncated denoising (not full diffusion, very fast)
     → Multi-mode trajectory distribution instead of single regression

Why this is strong:
- Your BEV tokens + diffusion head = multi-modal trajectory prediction
- 10× faster than standard diffusion, 45 FPS on RTX 4090 → trivially fast on L40S
- 64% better mode diversity — crucial for rare scenarios
- Pure drop-in: replace LightTrajectoryHead with the diffusion head, keep everything else

Hardware: Trivially fits, adds ~5-10% training cost

---
Direction 3: SparseDrive-style Sparse Perception + Planning

Based on: SparseDrive (arXiv:2405.19620), SOTA on nuScenes

Architecture:
Multi-view images (B, V, 3, H, W)
  └─ Backbone (DINOv2-Small or SigLIP, partially unfrozen)
  └─ Sparse 3D instance queries (not dense BEV!) — cross-attend to image features
  └─ Parallel streams:
     - Detection/tracking sparse queries
     - Map element sparse queries
     - Planning sparse queries (collision-aware rescoring)
  └─ Sparse planner head → waypoints

Why this is strong:
- No dense BEV → much more memory efficient, better spatial resolution
- Collision-aware rescoring filters unsafe trajectories explicitly
- All tasks (detect, track, map, plan) share the same sparse representation
- Open-source reference implementation available

Hardware: More memory efficient than dense BEV, fits well on 4× L40S

---
Direction 4: Hybrid Slow-Fast System (DriveVLM-Dual style)

Architecture:
"Fast path" (every frame, real-time):
  Your existing lightweight pipeline (Vision → Q-Former → BEV → Planner)

"Slow path" (every N frames, ~500ms latency OK):
  Qwen2.5-VL-7B with CoT:
    scene description → critical object identification → hierarchical plan
  → High-level intent override / trajectory correction for slow path

Fusion: slow path corrects fast path trajectory when confidence is low

Why this is elegant:
- You keep your existing trainable pipeline as the "fast brain"
- Add VLM reasoning only for hard cases (construction zones, pedestrians, cyclists)
- Mirrors how DriveVLM-Dual got 0.31m L2 on nuScenes (SOTA at time)
- Inference: fast path on 1 GPU, VLM on another GPU in parallel

---
My Concrete Recommendation: A 3-Phase Plan

Phase 1: Fix the Foundation (1-2 weeks)

Before adding VLMs, fix what's broken in your current architecture:

1. Unfreeze the backbone partially — unfreeze the last 4 transformer blocks of CLIP/DINOv2. Full
frozen backbone = no learning from driving data.
2. Replace your BEV 16×16 with higher resolution — try 32×32 or use sparse BEV queries (à la
BEVFormer's deformable attention)
3. Add real temporal fusion — stack 3-5 frames, use temporal cross-attention not just positional
embeddings
4. Fix the loss — add a heading/yaw loss, smooth velocity consistency loss, and a collision proxy
loss (penalize waypoints that overlap with detected objects)

Phase 2: DiffusionDrive Head (2-3 weeks)

Replace LightTrajectoryHead with a truncated diffusion decoder:
- Generate K=6 trajectory candidates from pre-computed anchors
- 2-step denoising
- Score candidates and return weighted ensemble
- Train with NLL loss on trajectory distribution, not just ADE/FDE

Phase 3: Qwen2.5-VL Integration + RL (4-6 weeks)

- Fine-tune Qwen2.5-VL-7B with LoRA on Waymo WOD-E2E data (SFT phase)
- Use Qwen2.5-VL-72B (via API or 4-bit QLoRA) to auto-generate CoT annotations ("why did the car
turn here?")
- GRPO RL fine-tuning: reward = RFS-proxy score (ADE on hard scenarios only, collision penalty)
- Either use standalone or as the "slow path" in a hybrid system

---
Key Papers to Read First

1. AutoVLA — arXiv:2506.13757 — closest to winning approach on Waymo, open-source
2. DiffusionDrive — arXiv:2411.15139 — best planner head, trivial to integrate
3. SparseDrive — arXiv:2405.19620 — best camera-only perception backbone
4. "Is Ego Status All You Need?" — arXiv:2312.03031 — critical: explains why nuScenes benchmarks
are misleading, validates that Waymo WOD-E2E is the right target
5. WOD-E2E Dataset paper — arXiv:2510.26125 — understand the RFS metric deeply before training

---
Bottom Line

Your current architecture is a solid scaffold but is fundamentally limited by:
- Frozen backbone (no domain adaptation)
- Dense coarse BEV (too lossy)
- ADE/FDE regression loss (wrong signal for WOD-E2E)
- No reasoning for long-tail scenarios

The fastest path to competitive WOD-E2E results: AutoVLA pattern with Qwen2.5-VL-7B + LoRA + GRPO
RL. You already have Qwen working in motion/qwen.py. The VLM's ability to reason about unusual
scenarios is precisely what WOD-E2E is designed to test.

---
Next: Phase 2 and Phase 3

Once you've run a training sweep and validated loss curves, swap in the DiffusionDrive head before
moving to the Qwen2.5-VL backbone in Phase 3. Since you've worked with AutoVLA, Phase 3 will
mostly be adapting the Qwen2.5-VL-3B/7B fine-tuning loop to your Waymo data format.

---
Code review findings + what was built

Bug fixed in `pipeline.py`

`BEVFeatureEncoder.forward` expects `(B, N, D)` but was receiving `(B, T, N, D)` when
`num_temporal_frames > 1`. Fixed by running `TemporalCrossAttention` first (which collapses T
frames → single set of tokens), then piping the result into `BEVFeatureEncoder`. The order
is now semantically correct too — BEV is built from temporally-fused features.

---
3 new files for AutoVLA direction

`waymo_e2e/prompts/waymo_prompt.py`
- `SYSTEM_PROMPT` — standard driving planner system context
- `build_user_prompt(intent, speed_mps)` — converts Waymo intent int + estimated speed to
  natural language
- `estimate_speed_mps(past_xy)` — estimates current speed from the last two valid past
  positions (Waymo = 0.5s between states)
- `waypoints_to_text` / `parse_waypoints_from_text` — for SFT target construction and CoT
  generation decoding

`waymo_e2e/models/qwen_planner.py`
- `QwenPlannerConfig` — model_name, LoRA rank/alpha/dropout/targets, 4-bit/8-bit quant
  options, image_size
- `QwenE2EPlanner` — loads Qwen2.5-VL + applies LoRA via PEFT, adds regression head (MLP
  with LayerNorm + small-init on final layer for stable training start)
- `forward(inputs)` — fast mode: last-real-token hidden state → regression head → `(B, 20, 3)`
  waypoints (uses attention mask to find last non-padding token correctly)
- `generate_trajectory_text(inputs)` — slow/CoT mode for inference/annotation
- `prepare_inputs(images_bvhwc, intents, past_xy, device)` — numpy→PIL + processor call,
  handles full batch

`train_autovla.py`
- Uses **accelerate** (correct tool for LoRA + large model, vs DataParallel which doesn't
  handle weight-shared models well)
- bf16 mixed precision built-in
- Differential LR: LoRA adapters get LR × 0.1, regression head gets full LR=2e-4
- Gradient accumulation = 4 → effective batch = 32 on 4 GPUs
- Saves only LoRA adapters + regression head (not full 7B weights) every 2 epochs

---
How to run (AutoVLA SFT)

```bash
# SFT with 4 GPUs (run accelerate config first if not done)
accelerate config   # one-time setup
accelerate launch --num_processes 4 train_autovla.py

# Single GPU for testing
python train_autovla.py
```

Next step: once you have a few epochs of SFT loss converging, we can build
`train_autovla_rl.py` with GRPO for Phase 2 (RL against RFS-proxy).

