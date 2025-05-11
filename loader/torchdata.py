import os
import sys
import math

import tensorflow as tf
import torch
import gcsfs
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset

# === Set up path to Waymo SDK ===
sys.path.append(os.path.join(os.path.dirname(__file__), 'waymo-open-dataset', 'src'))

# === Intent Label Mapping ===
intent_dict = {
    "UNKNOWN": 0,
    "GO_STRAIGHT": 1,
    "GO_LEFT": 2,
    "GO_RIGHT": 3
}

# === Pose embedding and Intent Embedding (variable-length support) ===
pose_mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 3)),  # variable time steps, 3 features
    tf.keras.layers.GlobalAveragePooling1D(), # aggregate over time
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32)
])
intent_embedding = tf.keras.layers.Embedding(input_dim=5, output_dim=16)

# === Parse & preprocess one record ===
def decode_and_preprocess(raw_bytes):
    def _parse(record_bytes):
        frame = wod_e2ed_pb2.E2EDFrame()
        frame.ParseFromString(record_bytes.numpy())

        # -- Images --
        camera_order = [
            "FRONT_LEFT","FRONT","FRONT_RIGHT",
            "SIDE_LEFT","SIDE_RIGHT",
            "REAR_RIGHT","REAR","REAR_LEFT"
        ]
        image_map = {open_dataset.CameraName.Name.Name(img.name): img for img in frame.frame.images}
        imgs = []
        for cam in camera_order:
            img = image_map[cam]
            raw = tf.image.decode_jpeg(img.image)
            raw = tf.image.resize(raw, (224, 224))
            raw = tf.cast(raw, tf.float32) / 255.0
            norm = (raw - tf.constant([0.485,0.456,0.406])) / tf.constant([0.229,0.224,0.225])
            norm = tf.transpose(norm, [2,0,1])
            imgs.append(norm)
        images = tf.stack(imgs)  # [8,3,224,224]

        # -- Pose embedding --
        past = frame.past_states
        N = len(past.pos_x)
        seq = []
        for i in range(N):
            vx, vy = past.vel_x[i], past.vel_y[i]
            hdg = math.atan2(vy, vx)
            seq.append([past.pos_x[i], past.pos_y[i], hdg])
        pose_tensor = tf.convert_to_tensor([seq], dtype=tf.float32)  # shape [1, T, 3]
        pose_emb = pose_mlp(pose_tensor)[0]  # [32]

        # -- Intent embedding --
        intent_enum = wod_e2ed_pb2.E2EDFrame.DESCRIPTOR.fields_by_name['intent'].enum_type.values_by_number[frame.intent]
        intent_id = intent_dict.get(intent_enum.name, 0)
        route_emb = intent_embedding(tf.constant([intent_id], dtype=tf.int32))[0]  # [16]

        return images.numpy(), pose_emb.numpy(), route_emb.numpy()

    imgs_np, pose_np, route_np = tf.py_function(_parse, [raw_bytes], [tf.float32, tf.float32, tf.float32])
    imgs_np.set_shape([8, 3, 224, 224])
    pose_np.set_shape([32])
    route_np.set_shape([16])
    return imgs_np, pose_np, route_np

# === Build streaming TFRecord dataset ===
def build_streaming_dataset(file_list, batch_size=4):
    ds = tf.data.TFRecordDataset(file_list)
    ds = ds.map(decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# === Streamer class ===
class WaymoTFRecordStreamer:
    def __init__(self, gcs_pattern, batch_size=4, token='google_default'):
        # Discover files directly with glob pattern
        self.fs = gcsfs.GCSFileSystem(token=token)
        self.tf_files = sorted(self.fs.glob(gcs_pattern))
        #print(f"[INFO] Found {len(self.tf_files)} TFRecord files to stream.")
        self.batch_size = batch_size
        self.dataset = build_streaming_dataset(self.tf_files, batch_size)

    def stream_to_torch(self):
        total_instances = 0
        for batch_idx, (img_tf, pose_tf, route_tf) in enumerate(self.dataset):
            imgs_t = torch.from_numpy(img_tf.numpy())    # [B,8,3,224,224]
            pose_t = torch.from_numpy(pose_tf.numpy())    # [B,32]
            route_t = torch.from_numpy(route_tf.numpy())  # [B,16]
            total_instances += imgs_t.shape[0]
        #print(f"[INFO] Completed streaming & conversion. Total instances: {total_instances}")

# === Main ===
if __name__ == "__main__":
    # Use correct glob to match all 263 TFRecord shards
    pattern = 'gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training_*.tfrecord*'
    streamer = WaymoTFRecordStreamer(pattern, batch_size=4)
    streamer.stream_to_torch()
