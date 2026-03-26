"""
AutoVLA-style E2E trajectory planner using Qwen2.5-VL as the vision-language backbone.

Architecture
------------
    Multi-view camera images (numpy, [0,1])
        + text prompt (navigation command + ego speed)
    ──────────────────────────────────────────────
    Qwen2.5-VL processor  →  tokenised input_ids + pixel_values
    Qwen2.5-VL forward    →  last-real-token hidden state (B × D)
    Regression head       →  (B × num_waypoints × 3) waypoints

Two forward modes
-----------------
  fast  (training + real-time inference):
        regression head on last hidden state → waypoints directly
  slow  (CoT inference / annotation):
        autoregressive text generation → parse_waypoints_from_text

Training phases
---------------
  Phase 1 (SFT):  trajectory_loss on regression head output
  Phase 2 (RL):   GRPO with RFS-proxy reward  (future train_autovla_rl.py)

Hardware
--------
  Qwen2.5-VL-7B in bfloat16: ~14 GB per GPU.
  LoRA (r=16):               fits easily on a single L40S 40 GB.
  4× L40S with DDP:          effective batch ~32 with grad-accum=4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QwenPlannerConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Modules to apply LoRA to. Covers both attention and FFN for best coverage.
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Quantisation (mutually exclusive)
    load_in_4bit: bool = False   # QLoRA; saves ~50% VRAM vs bfloat16
    load_in_8bit: bool = False

    torch_dtype: str = "bfloat16"   # "float16" on older GPUs

    # Regression head
    num_waypoints: int = 20
    regress_hidden_dim: int = 1024

    # Vision processor
    # Qwen2.5-VL handles dynamic resolution internally; 448 is a good trade-off
    # between quality and speed for driving cameras.
    image_size: int = 448


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QwenE2EPlanner(nn.Module):
    """
    Qwen2.5-VL with LoRA fine-tuning + a lightweight regression head for
    trajectory prediction.

    The VLM processes V multi-view images and a text prompt and returns the
    hidden state of the last real (non-padding) token.  The regression head
    maps this to (num_waypoints, 3) waypoints.

    Args:
        config: QwenPlannerConfig
    """

    def __init__(self, config: QwenPlannerConfig | None = None):
        super().__init__()
        self.config = config or QwenPlannerConfig()
        c = self.config
        dtype = torch.bfloat16 if c.torch_dtype == "bfloat16" else torch.float16

        # --- Processor (image + text tokeniser) ---
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            c.model_name, trust_remote_code=True
        )

        # --- Base VLM ---
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_kwargs: dict = dict(trust_remote_code=True, torch_dtype=dtype)

        if c.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif c.load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            c.model_name, **model_kwargs
        )

        # --- LoRA via PEFT ---
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=c.lora_rank,
            lora_alpha=c.lora_alpha,
            lora_dropout=c.lora_dropout,
            target_modules=c.lora_target_modules,
            bias="none",
        )
        self.vlm = get_peft_model(base, lora_cfg)
        self.vlm.print_trainable_parameters()

        hidden_size = base.config.hidden_size

        # --- Regression head (fast-mode planning) ---
        self.regress_head = nn.Sequential(
            nn.Linear(hidden_size, c.regress_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(c.regress_hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(c.regress_hidden_dim, c.num_waypoints * 3),
        )
        # Initialise last linear to near-zero so training starts from small predictions
        nn.init.normal_(self.regress_head[-1].weight, std=0.01)
        nn.init.zeros_(self.regress_head[-1].bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _last_real_hidden(self, inputs: dict) -> torch.Tensor:
        """
        Run VLM forward and return the hidden state of the last real
        (non-padding) token for each sample: (B, D).
        """
        out = self.vlm(**inputs, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1]          # (B, S, D)
        attn_mask = inputs["attention_mask"]    # (B, S)
        # Index of the last attended token per sample
        seq_lens = attn_mask.sum(dim=1) - 1    # (B,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, seq_lens]      # (B, D)

    # ------------------------------------------------------------------
    # Forward (fast mode — used during training)
    # ------------------------------------------------------------------

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        Fast-mode forward.

        Args:
            inputs: dict from ``prepare_inputs`` (already on correct device).

        Returns:
            waypoints: (B, num_waypoints, 3) float tensor
        """
        h = self._last_real_hidden(inputs)
        # Cast to regression head dtype in case of mixed precision
        h = h.to(self.regress_head[0].weight.dtype)
        return self.regress_head(h).reshape(h.size(0), self.config.num_waypoints, 3)

    # ------------------------------------------------------------------
    # Slow-mode inference (CoT text generation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_trajectory_text(
        self,
        inputs: dict,
        max_new_tokens: int = 300,
    ) -> list[str]:
        """
        Slow / CoT mode: generate waypoints as free-form text.

        Returns:
            List of generated strings, one per sample.
        """
        prompt_len = inputs["input_ids"].shape[1]
        generated = self.vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        images_bvhwc: np.ndarray,
        intents: np.ndarray,
        past_xy: np.ndarray | None = None,
        device: torch.device | str = "cuda",
        system_prompt: str | None = None,
    ) -> dict:
        """
        Convert a batch of numpy arrays into VLM processor inputs.

        Args:
            images_bvhwc:  (B, V, H, W, 3) float32 in [0, 1].
            intents:       (B,)  int32 Waymo intent codes (0–3).
            past_xy:       (B, P, 2) float32 past ego positions (for speed estimate).
                           Pass None to omit speed from the prompt.
            device:        Target device for returned tensors.
            system_prompt: Override the default SYSTEM_PROMPT.

        Returns:
            dict of tensors ready for ``self.forward()`` or ``generate_trajectory_text()``.
        """
        from qwen_vl_utils import process_vision_info
        from waymo_e2e.prompts.waymo_prompt import (
            SYSTEM_PROMPT,
            build_user_prompt,
            estimate_speed_mps,
        )

        B, V = images_bvhwc.shape[:2]
        sys_p = system_prompt or SYSTEM_PROMPT

        messages_batch = []
        for b in range(B):
            # -- Speed estimate from past trajectory --
            speed = None
            if past_xy is not None:
                speed = estimate_speed_mps(past_xy[b])  # float | None

            # -- PIL images for each view --
            content: list[dict] = []
            for v in range(V):
                arr = (images_bvhwc[b, v] * 255.0).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr, mode="RGB")
                content.append({"type": "image", "image": pil})

            # -- Text prompt --
            content.append({
                "type": "text",
                "text": build_user_prompt(int(intents[b]), speed_mps=speed),
            })

            messages_batch.append([
                {"role": "system", "content": sys_p},
                {"role": "user",   "content": content},
            ])

        # -- Apply chat template to get text strings --
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_batch
        ]

        # -- Extract image tensors (flat list across all samples) --
        all_images: list = []
        for msgs in messages_batch:
            img_inputs, _ = process_vision_info(msgs)
            if img_inputs:
                all_images.extend(img_inputs)

        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in inputs.items()}
