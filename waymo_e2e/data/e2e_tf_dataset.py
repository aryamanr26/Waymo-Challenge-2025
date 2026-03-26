"""TensorFlow tf.data pipeline for Waymo E2EDFrame records."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import torch
from io import BytesIO

from PIL import Image
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

# Fixed past length for tf.data batching + collision proxy loss
_PAST_XY_PAD = 64

from waymo_e2e.data.gcs_loader import WaymoDatasetLoader


class WaymoE2EDataset:
    """Waymo End-to-End Dataset Loader for TensorFlow-based input pipelines."""

    INTENT_MAP = {
        1: "GO STRAIGHT",
        2: "GO LEFT",
        3: "GO RIGHT",
    }

    def __init__(
        self,
        img_size=(224, 224),
        batch_size=8,
        shuffle_buffer=100,
        num_temporal_frames: int = 1,
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.num_temporal_frames = int(num_temporal_frames)

    def count_records_in_tfrecords(self, file_paths, compression_type=""):
        """Count E2EDFrame records in TFRecord files."""
        total_count = 0
        for i, file in enumerate(file_paths):
            count = 0
            dataset = tf.data.TFRecordDataset(file, compression_type=compression_type)
            for idx, raw_record in enumerate(dataset):
                data = wod_e2ed_pb2.E2EDFrame()
                data.ParseFromString(raw_record.numpy())
                intent_val = data.intent
                intent_str = self.INTENT_MAP.get(intent_val, f"unknown ({intent_val})")
                if idx % 100 == 0:
                    print(f"[Record {idx}] Intent: {intent_str}")
                count += 1
            print(f"[{i + 1}/{len(file_paths)}] {file}: {count} E2EDFrame instances")
            total_count += count
        return total_count

    def _decode_e2edframe_bytes(self, record_bytes: bytes):
        """
        Decode one E2EDFrame from raw bytes.
        Returns numpy arrays: images (V,H,W,3), intent (scalar int32), past_xy (P,2),
        future_xyz (F,3), pose_token (64,), routing_token (16,).
        """
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(record_bytes)

        past_raw = np.asarray(
            list(zip(data.past_states.pos_x, data.past_states.pos_y)), dtype=np.float32
        )
        past_states_xy = self._pad_past_xy(past_raw)

        future_states_xyz = np.asarray(
            list(zip(data.future_states.pos_x, data.future_states.pos_y, data.future_states.pos_z)),
            dtype=np.float32,
        )

        image_list = []
        for image_content in data.frame.images:
            img_bytes = bytes(image_content.image)
            try:
                im = Image.open(BytesIO(img_bytes)).convert("RGB")
                im = im.resize((self.img_size[1], self.img_size[0]), Image.BICUBIC)
                rgb = np.asarray(im, dtype=np.float32) / 255.0
            except (OSError, ValueError):
                rgb = np.zeros((*self.img_size, 3), dtype=np.float32)
            image_list.append(rgb)

        while len(image_list) < 3:
            image_list.append(np.zeros((*self.img_size, 3), dtype=np.float32))

        images = np.stack(image_list, axis=0)
        intent = np.int32(data.intent)

        pose_flat = past_raw.reshape(-1)[:64]
        pose_token = np.pad(pose_flat, (0, max(0, 64 - pose_flat.size))).astype(np.float32)

        intent_onehot = np.zeros(4, dtype=np.float32)
        if 0 <= int(intent) < 4:
            intent_onehot[int(intent)] = 1.0
        routing_token = np.pad(intent_onehot, (0, 12)).astype(np.float32)

        return images, intent, past_states_xy, future_states_xyz, pose_token, routing_token

    @staticmethod
    def _pad_past_xy(past_xy: np.ndarray) -> np.ndarray:
        """Pad / trim past xy to (_PAST_XY_PAD, 2) for dense batching."""
        out = np.zeros((_PAST_XY_PAD, 2), dtype=np.float32)
        if past_xy.size == 0:
            return out
        n = min(past_xy.shape[0], _PAST_XY_PAD)
        out[-n:] = past_xy[-n:]
        return out

    def parse_e2edframe(self, record):
        """Parse a single serialized E2EDFrame from TFRecord (TensorFlow path, JPEG decode)."""
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(record.numpy())

        past_raw = tf.convert_to_tensor(
            list(zip(data.past_states.pos_x, data.past_states.pos_y)), dtype=tf.float32
        )
        past_states_xy = past_raw[-_PAST_XY_PAD:]
        pad_rows = _PAST_XY_PAD - tf.shape(past_states_xy)[0]
        past_states_xy = tf.pad(past_states_xy, [[pad_rows, 0], [0, 0]])

        pose_flat = tf.reshape(past_raw, [-1])[:64]
        pad_pose = tf.maximum(0, 64 - tf.shape(pose_flat)[0])
        pose_token = tf.pad(pose_flat, [[0, pad_pose]])

        future_states_xyz = tf.convert_to_tensor(
            list(zip(data.future_states.pos_x, data.future_states.pos_y, data.future_states.pos_z)),
            dtype=tf.float32,
        )

        image_list = []
        for image_content in data.frame.images:
            img_tensor = tf.io.decode_jpeg(image_content.image, channels=3)
            img_tensor = tf.image.resize(img_tensor, self.img_size)
            img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
            image_list.append(img_tensor)

        while len(image_list) < 3:
            image_list.append(tf.zeros([*self.img_size, 3], dtype=tf.float32))

        images = tf.stack(image_list)
        intent = tf.convert_to_tensor(data.intent, dtype=tf.int32)

        intent_onehot = tf.one_hot(intent, depth=4)
        routing_token = tf.pad(intent_onehot, [[0, 12]])

        return images, intent, past_states_xy, future_states_xyz, pose_token, routing_token

    def tf_parse_wrapper(self):
        def wrapped(record):
            out = tf.py_function(
                func=self.parse_e2edframe,
                inp=[record],
                Tout=(tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32),
            )
            out[0].set_shape([3, *self.img_size, 3])
            out[2].set_shape([_PAST_XY_PAD, 2])
            return out

        return wrapped

    def _parse_e2edframe_window_np(self, records):
        """Stack ``num_temporal_frames`` consecutive frames; labels from the last frame."""
        T = self.num_temporal_frames
        buf = records.numpy()
        imgs_stack = []
        for i in range(T):
            imgs, _, _, _, _, _ = self._decode_e2edframe_bytes(bytes(buf[i]))
            imgs_stack.append(imgs)
        stacked = np.stack(imgs_stack, axis=0).astype(np.float32)
        _, intent, past_xy, fut, pose_tok, route_tok = self._decode_e2edframe_bytes(bytes(buf[-1]))
        return (
            stacked,
            np.int32(intent),
            past_xy.astype(np.float32),
            fut.astype(np.float32),
            pose_tok.astype(np.float32),
            route_tok.astype(np.float32),
        )

    def tf_parse_window_wrapper(self):
        T = self.num_temporal_frames

        def wrapped(records):
            out = tf.py_function(
                func=self._parse_e2edframe_window_np,
                inp=[records],
                Tout=(tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32),
            )
            out[0].set_shape([T, 3, *self.img_size, 3])
            out[1].set_shape([])
            out[2].set_shape([_PAST_XY_PAD, 2])
            out[3].set_shape([None, 3])
            out[4].set_shape([64])
            out[5].set_shape([16])
            return out

        return wrapped

    def build_dataset(self, file_paths):
        dataset = tf.data.TFRecordDataset(file_paths)
        if self.num_temporal_frames <= 1:
            dataset = dataset.map(self.tf_parse_wrapper(), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            T = self.num_temporal_frames
            dataset = dataset.window(T, shift=1, drop_remainder=True)
            dataset = dataset.flat_map(lambda w: w.batch(T))
            dataset = dataset.map(self.tf_parse_window_wrapper(), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()

    dataset_builder = WaymoE2EDataset(batch_size=8)
    train_ds = dataset_builder.build_dataset(train_files[:2])

    INTENT_MAP = WaymoE2EDataset.INTENT_MAP

    for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(1):
        images = torch.from_numpy(images.numpy())
        intent = torch.from_numpy(intent.numpy())
        past_states = torch.from_numpy(past_states.numpy())
        future_states = torch.from_numpy(future_states.numpy())
        pose_token = torch.from_numpy(pose_token.numpy())
        routing_token = torch.from_numpy(routing_token.numpy())

        print("First image shape:", images[0].shape)
        print("Image batch shape:", images.shape)
        print("Intent:", intent[0].item(), "->", INTENT_MAP.get(int(intent[0]), "Unknown"))
        print("Past states shape:", past_states[0].shape)
        print("Future states shape:", future_states[0].shape)
        print("Pose token shape:", pose_token.shape)
        print("Routing token shape:", routing_token.shape)
