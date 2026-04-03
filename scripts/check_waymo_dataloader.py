#!/usr/bin/env python3
"""
Smoke test: list Waymo E2E TFRecord paths (GCS) and read one batch via tf.data.

Run from repo root (or use scripts/check_waymo_dataloader.sh).

Exit codes: 0 = OK, 1 = failure (no files or read error).
"""

from __future__ import annotations

import os
import sys

# TensorFlow on CPU for this check (avoids GPU memory / CUDA coupling)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Repo root on sys.path when invoked as python scripts/check_waymo_dataloader.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    def log(msg: str) -> None:
        print(msg, flush=True)

    log("Waymo dataloader smoke test")
    log("  repo: " + _REPO_ROOT)
    log("  (importing TensorFlow / deps — can take 1–2 minutes on cold start)")
    try:
        from waymo_e2e.data import WaymoDatasetLoader, WaymoE2EDataset
    except ModuleNotFoundError as e:
        print("ERROR: missing python package dependency.")
        print(f"  {type(e).__name__}: {e}")
        print("Hint: install waymo-open-dataset in this env, or clone")
        print("  <repo>/waymo-open-dataset and export PYTHONPATH to its src/ directory.")
        return 1

    try:
        loader = WaymoDatasetLoader()
        train_files, val_files, test_files = loader.get_file_lists()
    except Exception as e:
        print("ERROR: WaymoDatasetLoader failed (GCS auth / network / gcsfs).")
        print(f"  {type(e).__name__}: {e}")
        return 1

    log(f"  listed: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    if not train_files:
        print("ERROR: no training TFRecords found.")
        return 1

    # Optional: WAYMO_NUM_TEMPORAL_FRAMES=5 to test windowed pipeline
    nt = int(os.environ.get("WAYMO_NUM_TEMPORAL_FRAMES", "1"))
    batch_size = int(os.environ.get("WAYMO_SMOKE_BATCH_SIZE", "2"))
    num_shards = int(os.environ.get("WAYMO_SMOKE_SHARDS", "1"))

    paths = train_files[:num_shards]
    log(
        f"  building dataset: num_temporal_frames={nt} batch_size={batch_size} shards={len(paths)}"
    )
    log(
        "  reading first batch from GCS (large shard + JPEG decode — often several minutes) ..."
    )

    try:
        builder = WaymoE2EDataset(
            batch_size=batch_size,
            num_temporal_frames=nt,
        )
        ds = builder.build_dataset(paths)
    except Exception as e:
        print("ERROR: WaymoE2EDataset.build_dataset failed.")
        print(f"  {type(e).__name__}: {e}")
        return 1

    try:
        for batch in ds.take(1):
            images, intent, past_states, future_states, pose_token, routing_token = batch
            log("  OK — first batch read successfully.")
            log("    images         " + str(images.shape) + " " + images.dtype.name)
            log("    intent         " + str(intent.shape) + " " + intent.dtype.name)
            log("    past_states    " + str(past_states.shape) + " " + past_states.dtype.name)
            log("    future_states  " + str(future_states.shape) + " " + future_states.dtype.name)
            log("    pose_token     " + str(pose_token.shape) + " " + pose_token.dtype.name)
            log("    routing_token  " + str(routing_token.shape) + " " + routing_token.dtype.name)
            log("PASSED.")
            return 0
    except Exception as e:
        print("ERROR: failed to iterate first batch (TF decode / proto / data).")
        print(f"  {type(e).__name__}: {e}")
        return 1

    print("ERROR: dataset produced zero batches.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
