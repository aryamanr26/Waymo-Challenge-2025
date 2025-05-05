import tensorflow as tf
import os
from waymo_dataset_loader import WaymoDatasetLoader

# ========== 1. Decode JPEGs from TFRecord Example ==========
def parse_waymo_example(record_bytes):
    # Parse the tf.train.Example from raw bytes
    example = tf.train.Example()
    example.ParseFromString(record_bytes.numpy())  # Pure Python op

    decoded_images = []
    for key, feature in example.features.feature.items():
        if feature.HasField('bytes_list'):
            values = feature.bytes_list.value
            if values and values[0][:3] == b'\xff\xd8\xff':  # JPEG magic number
                try:
                    img_tensor = tf.image.decode_jpeg(values[0])
                    decoded_images.append(img_tensor)
                except Exception as e:
                    print(f"Failed to decode {key}: {e}")
    
    return decoded_images

# Wrap parse logic to be usable in tf.data
def tf_parse_waymo_example(record):
    decoded_imgs = tf.py_function(func=parse_waymo_example, inp=[record], Tout=[tf.uint8])  # List of tensors
    return decoded_imgs

# ========== 2. Build Streaming Dataset from GCS ==========
def get_streaming_dataset(file_paths, batch_size=1):
    dataset = (
        tf.data.Dataset.from_tensor_slices(file_paths)
        .interleave(lambda fp: tf.data.TFRecordDataset(fp), cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE)
        .map(tf_parse_waymo_example, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset

# ========== 3. GCS File Discovery ==========
def list_gcs_tfrecords(bucket_path):
    # Assumes GCS credentials are set up via `gcloud auth application-default login`
    files = tf.io.gfile.listdir(bucket_path)
    return [os.path.join(bucket_path, fname) for fname in files if fname.endswith('.tfrecord')]

# ========== 4. Main ==========
if __name__ == "__main__":
    # BUCKET_PATH = "gs://your-bucket-name/waymo"
    # train_files = list_gcs_tfrecords(BUCKET_PATH)
    loader = WaymoDatasetLoader()
    train_files, val_files, test_files = loader.get_file_lists()

    print(f"Found {len(train_files)} training files.")
    
    # Stream and process
    dataset = get_streaming_dataset(train_files, batch_size=2)

    for i, batch in enumerate(dataset.take(5)):
        print(f"\nBatch {i+1}")
        for j, decoded_img_set in enumerate(batch):  # batch is a list of [8 images]
            print(f"  Record {j+1}: {len(decoded_img_set)} images extracted")
