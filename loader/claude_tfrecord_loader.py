import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from waymo_dataset_loader import WaymoDatasetLoader

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 4  # Tune based on memory/GPU

def count_records_in_tfrecords(file_paths, compression_type=''):
    total_count = 0
    for i, file in enumerate(file_paths):
        dataset = tf.data.TFRecordDataset(file, compression_type=compression_type)
        count = sum(1 for _ in dataset)
        print(f"[{i+1}/{len(file_paths)}] {file}: {count} records")
        total_count += count
    print(f"\nTotal Records Across {len(file_paths)} Files: {total_count}")
    return total_count

# Python JPEG extractor (runs inside py_function)
def parse_waymo_example(record_bytes):
    example = tf.train.Example()
    example.ParseFromString(record_bytes.numpy())
    
    decoded_images = []
    for key, feature in example.features.feature.items():
        if feature.HasField('bytes_list'):
            values = feature.bytes_list.value
            for value in values:
                if value and len(value) > 3 and value[:3] == b'\xff\xd8\xff':  # JPEG magic bytes
                    try:
                        img_tensor = tf.image.decode_jpeg(value)
                        decoded_images.append(img_tensor)
                    except Exception as e:
                        print(f"Failed to decode {key}: {e}")
    
    if len(decoded_images) == 0:
        return tf.zeros([1, 64, 64, 3], dtype=tf.uint8)
    else:
        # Ensure all images have the same shape
        shapes = [img.shape for img in decoded_images]
        if not all(shape == shapes[0] for shape in shapes):
            # Resize all images to the same shape if necessary
            target_shape = decoded_images[0].shape[:2]  # (height, width)
            for i in range(1, len(decoded_images)):
                if decoded_images[i].shape[:2] != target_shape:
                    decoded_images[i] = tf.image.resize(decoded_images[i], target_shape)
                    decoded_images[i] = tf.cast(decoded_images[i], tf.uint8)
        
        return tf.stack(decoded_images)

# Wrapper for the parser in tf.data pipeline
@tf.autograph.experimental.do_not_convert
def tf_parse_waymo_example(record):
    images = tf.py_function(
        func=parse_waymo_example,
        inp=[record],
        Tout=tf.uint8
    )
    # We need to set a partially known shape, but can't know exact dimensions
    images.set_shape([None, None, None, 3])  # (num_images, H, W, 3)
    return images

# Build full TFRecord pipeline
def build_dataset(tfrecord_files, batch_size=BATCH_SIZE):
    raw_dataset = tf.data.TFRecordDataset(
        tfrecord_files, 
        compression_type='',  # Update if your TFRecords are compressed
        num_parallel_reads=AUTOTUNE
    )
    dataset = raw_dataset.map(tf_parse_waymo_example, num_parallel_calls=AUTOTUNE)
    
    # Filter out empty examples (optional)
    # dataset = dataset.filter(lambda x: tf.shape(x)[0] > 0)
    
    # Handle variable number of images per example
    # Must explicitly use uint8 type for padding values to match the output type
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=[None, None, None, 3],
        padding_values=tf.constant(0, dtype=tf.uint8)
    )
    
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

if __name__ == "__main__":
    # Make sure WaymoDatasetLoader is properly implemented
    loader = WaymoDatasetLoader()
    train_files, val_files, test_files = loader.get_file_lists()
    
    print("Training set size:", len(train_files))
    if train_files:
        print("First training file:", train_files[0])
        
        # Count records in the first file
        total = count_records_in_tfrecords(train_files[:1])
        print("Total records in first file:", total)
        
        print("\nBuilding streamed dataset pipeline...")
        dataset = build_dataset(train_files[:2])  # Just a couple files for testing
        
        # Display batch information and visualize
        for i, batch in enumerate(dataset.take(2)):
            print(f"\nBatch {i+1}")
            print(" Shape:", batch.shape)  # (B, N, H, W, 3)
            
            # Example: show first image of first record
            if batch.shape[0] > 0 and batch.shape[1] > 0:
                first_img = batch[0][0].numpy()
                plt.figure(figsize=(10, 8))
                plt.imshow(first_img)
                plt.title(f"Sample image from Batch {i+1}")
                plt.axis("off")
                plt.show()
            else:
                print("Batch contains no valid images.")
    else:
        print("No training files found. Check your WaymoDatasetLoader implementation.")