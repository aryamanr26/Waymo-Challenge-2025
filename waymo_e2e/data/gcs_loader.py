"""GCS file listing for Waymo E2E camera TFRecords.

This module intentionally avoids importing heavy Waymo TF custom ops at import-time.
Listing GCS file paths does not require camera ops/protos; eager imports can fail on
clusters with incompatible TensorFlow/absl binary combinations.
"""

import os
import sys

import gcsfs

# Keep local Waymo source discoverable for downstream scripts that may import it.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAYMO_SRC_PATH = os.path.join(_REPO_ROOT, "waymo-open-dataset", "src")
if WAYMO_SRC_PATH not in sys.path:
    sys.path.append(WAYMO_SRC_PATH)


class WaymoDatasetLoader:
    def __init__(self, bucket_name="waymo_open_dataset_end_to_end_camera_v_1_0_0", gcs_prefix="gs://"):
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.fs = gcsfs.GCSFileSystem(token="google_default")
        self.train_files = []
        self.val_files = []
        self.test_files = []

    def _list_gcs_files(self):
        print(f"Accessing GCS bucket: {self.bucket_name}")
        return self.fs.glob(f"{self.bucket_name}/*")

    def _filter_and_sort_files(self, all_files):
        self.train_files = [self.gcs_prefix + f for f in all_files if os.path.basename(f).startswith("training_")]
        self.val_files = [self.gcs_prefix + f for f in all_files if os.path.basename(f).startswith("val_")]
        self.test_files = [self.gcs_prefix + f for f in all_files if os.path.basename(f).startswith("test_")]
        self.train_files.sort()
        self.val_files.sort()
        self.test_files.sort()

    def load_dataset_file_paths(self):
        all_files = self._list_gcs_files()
        self._filter_and_sort_files(all_files)

    def get_file_lists(self):
        if not self.train_files:
            self.load_dataset_file_paths()
        return self.train_files, self.val_files, self.test_files
