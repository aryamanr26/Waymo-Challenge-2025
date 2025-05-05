import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_dataset_loader import WaymoDatasetLoader

''' Original Script for this object-oriented Waymo E2EDataset: load_dataset.py. If you have any changes to make, change it in that script then here.'''
class WaymoE2EDataset:
    INTENT_MAP = {
        1: "GO STRAIGHT",
        2: "GO LEFT",
        3: "GO RIGHT"
    }

    def __init__(self, img_size=(224, 224), batch_size=8, shuffle_buffer=100):
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    def count_records_in_tfrecords(self, file_paths, compression_type=''):
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
            print(f"[{i+1}/{len(file_paths)}] {file}: {count} E2EDFrame instances")
            total_count += count
        return total_count

    def parse_e2edframe(self, record):
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(record.numpy())
        image_list = []
        calibration_list = []

        past_states_xy = tf.convert_to_tensor(
            list(zip(data.past_states.pos_x, data.past_states.pos_y)), dtype=tf.float32
        )

        future_states_xyz = tf.convert_to_tensor(
            list(zip(data.future_states.pos_x, data.future_states.pos_y, data.future_states.pos_z)), dtype=tf.float32
        )

        order = [2, 1, 3]  # Front view cameras
        for index, image_content in enumerate(data.frame.images):
            if image_content.name in order:
                calibration = data.frame.context.camera_calibrations[index]
                img_tensor = tf.io.decode_jpeg(image_content.image, channels=3)
                img_tensor = tf.image.resize(img_tensor, self.img_size)
                img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
                image_list.append(img_tensor)
                calibration_list.append(calibration)

        while len(image_list) < 3:
            image_list.append(tf.zeros([*self.img_size, 3], dtype=tf.float32))

        images = tf.stack(image_list)  # Shape: (3, H, W, 3)
        intent = tf.convert_to_tensor(data.intent, dtype=tf.int32)

        pose_flat = tf.reshape(past_states_xy, [-1])
        pose_flat = tf.expand_dims(pose_flat, axis=0)
        pose_token = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64)
        ])(pose_flat)
        pose_token = tf.reshape(pose_token, [-1])

        routing_token = tf.keras.layers.Embedding(input_dim=4, output_dim=16)(intent)

        return images, intent, past_states_xy, future_states_xyz, pose_token, routing_token

    def tf_parse_wrapper(self):
        def wrapped(record):
            return tf.py_function(
                func=self.parse_e2edframe,
                inp=[record],
                Tout=(tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32)
            )
        return wrapped

    def build_dataset(self, file_paths):
        dataset = tf.data.TFRecordDataset(file_paths)
        dataset = dataset.map(self.tf_parse_wrapper(), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

if __name__ == "__main__":
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()

    dataset_builder = WaymoE2EDataset(batch_size=8)
    train_ds = dataset_builder.build_dataset(train_files[:2])

    for images, intent, past_states, future_states, pose_token, routing_token in train_ds.take(1):
        print("First image shape:", images[0].shape)
        print("Intent:", intent[0].numpy(), "->", dataset_builder.INTENT_MAP.get(int(intent[0]), "Unknown"))
        print("Past states shape:", past_states[0].shape)
        print("Future states shape:", future_states[0].shape)
        print("Pose token shape:", pose_token.shape)
        print("Routing token shape:", routing_token.shape)