"""
Prompt templates and waypoint serialization for AutoVLA-style Waymo E2E planning.

Usage:
    from waymo_e2e.prompts import build_user_prompt, waypoints_to_text, parse_waypoints_from_text

Prompt structure (per sample):
    [system]: SYSTEM_PROMPT
    [user]:   <image> × V cameras + build_user_prompt(intent, speed_mps)
    [assistant] (SFT target): waypoints_to_text(waypoints_array)
"""

from __future__ import annotations

import re
import numpy as np

# ---------------------------------------------------------------------------
# Intent vocabulary  (matches Waymo E2EDFrame.intent field values)
# ---------------------------------------------------------------------------
_INTENT_TEXT = {
    0: "proceed (direction unknown)",
    1: "go straight",
    2: "turn left",
    3: "turn right",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert autonomous vehicle trajectory planner. "
    "You receive multi-camera images from the ego vehicle and its current driving state. "
    "Your task is to predict the future trajectory as a sequence of waypoints in the "
    "ego-vehicle coordinate frame (x = forward, y = left, z = up), spaced 0.2 seconds apart."
)


def build_user_prompt(
    intent: int,
    speed_mps: float | None = None,
    num_waypoints: int = 20,
) -> str:
    """
    Build the text portion of the user message.

    Args:
        intent:       Integer from Waymo E2EDFrame (0–3).
        speed_mps:    Current ego speed in m/s (estimated from past states). None if unknown.
        num_waypoints: How many waypoints to request.
    """
    intent_str = _INTENT_TEXT.get(int(intent), "go straight")
    duration_s = num_waypoints * 0.2

    lines = [
        f"Navigation command: {intent_str}.",
    ]
    if speed_mps is not None:
        lines.append(f"Current speed: {speed_mps:.1f} m/s.")

    lines += [
        f"Predict the next {num_waypoints} waypoints covering {duration_s:.1f} seconds "
        f"(one waypoint every 0.2 s) in the ego-vehicle coordinate frame.",
        "Output format — a Python list of (x, y, z) tuples, no extra text:",
        "[(x1,y1,z1), (x2,y2,z2), ...]",
    ]
    return "\n".join(lines)


def estimate_speed_mps(past_xy: np.ndarray, dt: float = 0.5) -> float | None:
    """
    Estimate current speed from the last two valid (non-zero) past XY positions.

    Args:
        past_xy: (P, 2) array; zero rows are treated as padding.
        dt:      Time between consecutive past states in seconds (Waymo = 0.5 s at 2 Hz).
    """
    valid = np.any(past_xy != 0, axis=-1)  # (P,)
    valid_xy = past_xy[valid]
    if len(valid_xy) < 2:
        return None
    disp = np.linalg.norm(valid_xy[-1] - valid_xy[-2])
    return float(disp / dt)


# ---------------------------------------------------------------------------
# Waypoint ↔ text serialization
# ---------------------------------------------------------------------------

def waypoints_to_text(waypoints: np.ndarray) -> str:
    """
    Serialize a (T, 3) float32 waypoint array to the target text format.

    Example output: "[(0.12,0.00,0.00), (0.35,0.01,0.00), ...]"
    """
    parts = [f"({x:.3f},{y:.3f},{z:.3f})" for x, y, z in waypoints]
    return "[" + ", ".join(parts) + "]"


def parse_waypoints_from_text(text: str, num_waypoints: int = 20) -> np.ndarray:
    """
    Parse model-generated text back to a (num_waypoints, 3) float32 array.

    Robust to extra whitespace, newlines, and partial output — missing values
    are zero-padded. Returns zeros on total parse failure.
    """
    try:
        # Extract all numeric tokens (int or float, with optional sign)
        nums = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", text)
        coords = [float(n) for n in nums[: num_waypoints * 3]]
    except Exception:
        coords = []

    # Zero-pad to exactly num_waypoints * 3 values
    coords += [0.0] * max(0, num_waypoints * 3 - len(coords))
    return np.array(coords[: num_waypoints * 3], dtype=np.float32).reshape(num_waypoints, 3)
