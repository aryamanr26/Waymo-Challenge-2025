import tensorflow as tf
import numpy as np
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

# Map intent integers to strings
INTENT_MAP = {
    1: "GO STRAIGHT",
    2: "GO LEFT",
    3: "GO RIGHT"
}

'''
Intent: High Level Commands
Past States: Pose X,Y Vel X,Y Acc X,Y - 16 instances at 4HZ, means 4 second past data
Future States: Pose X,Y,Z - 20 instances at 4Hz, means 5 sec future data
'''

def count_records_in_tfrecords(file_paths, compression_type=''):
    total_count = 0

    for i, file in enumerate(file_paths):
        count = 0
        dataset = tf.data.TFRecordDataset(file, compression_type=compression_type)

        for idx, raw_record in enumerate(dataset):
            data = wod_e2ed_pb2.E2EDFrame()
            data.ParseFromString(raw_record.numpy())

            intent_val = data.intent
            intent_str = INTENT_MAP.get(intent_val, f"unknown ({intent_val})")

            if idx % 100 == 0:
                print(f"[Record {idx}] Intent: {intent_str}")

            count += 1

        print(f"[{i+1}/{len(file_paths)}] {file}: {count} E2EDFrame instances")
        total_count += count

    return total_count

def parse_e2edframe(record):
    data = wod_e2ed_pb2.E2EDFrame()
    data.ParseFromString(record.numpy())
    image_list = []
    calibration_list = []

    # Past and future state tensors
    past_states_xy = tf.convert_to_tensor(
        list(zip(data.past_states.pos_x, data.past_states.pos_y)), dtype=tf.float32
    )  # (16, 2) - 4 second data

    future_states_xyz = tf.convert_to_tensor(
        list(zip(data.future_states.pos_x, data.future_states.pos_y, data.future_states.pos_z)), dtype=tf.float32
    )  # (20, 3) - 5 second data

    # Past velocity and acceleration data (IF NEEDED!)
    past_states_vel_xy = None # list(zip(data.past_states.pos_x, data.past_states.pos_y))
    past_states_accel_xy = None #list(zip(data.past_states.pos_x, data.past_states.pos_y))

    order = [2, 1, 3]  # Only taking the front view images

    # for camera_name in order:
    for index, image_content in enumerate(data.frame.images):
        if image_content.name in order:
            calibration = data.frame.context.camera_calibrations[index]
            # image = tf.io.decode_image(image_content.image).numpy()
            img_tensor = tf.io.decode_jpeg(image_content.image, channels=3)
            img_tensor = tf.image.resize(img_tensor, [224, 224])
            img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
            image_list.append(img_tensor)
            calibration_list.append(calibration)

    # Extract vision + language data (customize based on your model needs)

    # If fewer than 3 images (due to missing cameras), pad with zeros
    while len(image_list) < 3:
        image_list.append(tf.zeros([224, 224, 3], dtype=tf.float32))

    images = tf.stack(image_list)  # Shape: (3, 224, 224, 3)
    intent = tf.convert_to_tensor(data.intent, dtype=tf.int32)

        # === Pose Embedding === (Can also use a RNN/GRU Network)
    pose_flat = tf.reshape(past_states_xy, [-1])  # (32,)
    pose_flat = tf.expand_dims(pose_flat, axis=0)  
    pose_token = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64)
    ])(pose_flat)
    pose_token = tf.reshape(pose_token, [-1])

    # rnn = tf.keras.layers.GRU(64)
    # pose_token = rnn(past_states_xy[None, ...])  # Add batch dim if needed


    # === Routing Embedding ===
    token_length = 16
    routing_token = tf.keras.layers.Embedding(input_dim=4, output_dim=token_length)(intent)

    return images, intent, past_states_xy, future_states_xyz, pose_token, routing_token

def tf_parse_wrapper(record):
    return tf.py_function(
        func=parse_e2edframe,
        inp=[record],
        Tout=(tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32)
    )

def build_dataset(file_paths, batch_size=8, shuffle_buffer=100):
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(tf_parse_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    loader = WaymoDatasetLoader()
    train_files, val_files, test_files = loader.get_file_lists()

    print("Training set size:", len(train_files))
    print("First training file:", train_files[0])
    print("Second training file:", train_files[1])

    # print(count_records_in_tfrecords(train_files[:2]))
    train_ds = build_dataset(train_files[:2], batch_size=8)
    
    print("Build Dataset worked!\n")

    # === Display first instance of all variables ===
    for id, batch in enumerate(train_ds.take(1)):
        images, intent, past_states, future_states, pose_token, routing_token = batch
        print("First image tensor shape:", images[0].shape)
        print("First image tensor (min/max):",
              tf.reduce_min(images[0]).numpy(), tf.reduce_max(images[0]).numpy())

        print("First intent:", intent[0].numpy(), "->",
              INTENT_MAP.get(int(intent[0]), "Unknown"))

        print("First past_states shape:", past_states[0].shape)
        # print("First past_states values:\n", past_states[0].numpy())

        print("First future_states shape:", future_states[0].shape)
        # print("First future_states values:\n", future_states[0].numpy())

        print("Pose token shape:", pose_token.shape)
        # print("Pose token example:\n", pose_token[0].numpy())
        print("Routing token shape:", routing_token.shape)
        print("Routing token example:\n", routing_token[0].numpy())
        

    # Optionally count examples in a single TFRecord file
    # total = count_records_in_tfrecords(train_files[:1])  # Count just one file for now
    # print("Total records in first file: ", total)
