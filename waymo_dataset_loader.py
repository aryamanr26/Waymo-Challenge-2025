# waymo_dataset_loader.py

import os
import sys
import gcsfs
import tensorflow as tf
import numpy as np
import cv2

# Print version information (optional)
print(f"tensorflow: {tf.__version__}")
print(f"numpy: {np.__version__}")
print(f"cv2 (OpenCV): {cv2.__version__}")

# Append the Waymo API path for protobuf and ops imports
WAYMO_SRC_PATH = os.path.join(os.path.dirname(__file__), 'waymo-open-dataset', 'src')
sys.path.append(WAYMO_SRC_PATH)

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2


class WaymoDatasetLoader:
    def __init__(self, bucket_name="waymo_open_dataset_end_to_end_camera_v_1_0_0", gcs_prefix="gs://"):
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.fs = gcsfs.GCSFileSystem(token='google_default')

        # These will be populated after loading
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
        # print(f"Train Files: {len(self.train_files)}")
        # print(f"Validation Files: {len(self.val_files)}")
        # print(f"Test Files: {len(self.test_files)}")

    def get_file_lists(self):
        if not self.train_files:
            self.load_dataset_file_paths()
        return self.train_files, self.val_files, self.test_files


# For local testing
if __name__ == "__main__":
    loader = WaymoDatasetLoader()
    loader.load_dataset_file_paths()

    # print("\nSample training files:")
    # for path in loader.train_files[:5]:
    #     print(path)
