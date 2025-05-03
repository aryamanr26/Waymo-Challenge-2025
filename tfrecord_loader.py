import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from pprint import pprint
# import io
from waymo_dataset_loader import WaymoDatasetLoader

def extract_image_from_waymo_endtoend(file_path):
    if not tf.io.gfile.exists(file_path):
        print(f"Error: File {file_path} not found")
        return
    
    try:
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        
        for i, raw_data in enumerate(dataset.take(3)):
            #print(f"\nProcessing record {i+1}...")
            example = tf.train.Example()
            example.ParseFromString(raw_data.numpy())
            feature_keys = list(example.features.feature.keys())
            # print(f"Found {len(feature_keys)} features:")
            
            for key in feature_keys:
                feature = example.features.feature[key]
                
                if feature.HasField('bytes_list'):
                    values = feature.bytes_list.value
                    # print(f"  {key}: bytes[{len(values)}]")
                    
                    if len(values) > 0 and key not in ['state/pose']:
                        first_bytes = values[0][:20]
                        # print(f"    First bytes: {first_bytes.hex()}")
                        
                        if first_bytes.startswith(b'\xff\xd8\xff'):
                            # print(f"    *** Possible JPEG image in {key} ***")
                            
                            try:
                                img = tf.image.decode_jpeg(values[0]).numpy()
                                print(f"    Successfully decoded image: shape={img.shape}")
                                
                                # plt.figure(figsize=(12, 8))
                                # plt.imshow(img)
                                # plt.title(f"Image from feature: {key}")
                                # plt.axis("off")
                                # plt.show()
                            except Exception as e:
                                print(f"    Failed to decode as image: {e}")
                
                elif feature.HasField('float_list'):
                    values = feature.float_list.value
                    # print(f"  {key}: float[{len(values)}]")
                    if len(values) < 10:
                        # print(f"    Values: {list(values)}")
                        pass
                        
                elif feature.HasField('int64_list'):
                    values = feature.int64_list.value
                    # print(f"  {key}: int64[{len(values)}]")
                    if len(values) < 10:
                        # print(f"    Values: {list(values)}")
                        pass
    
    except Exception as e:
        print(f"Error processing dataset: {type(e).__name__}: {e}")

def try_extract_common_formats(file_path):
    # print("Trying various extraction methods...")
    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    print("Tf dataset craeated!!")
    # print(dataset)
    # for raw_data in dataset.take(1):
    for raw_data in dataset:
        serial = list() ## Stores all 8 camera images in one instance
        data = raw_data.numpy()
        jpeg_found = False
        start_pos = 0
        
        while True:
            jpeg_start = data.find(b'\xff\xd8\xff', start_pos)
            if jpeg_start == -1:
                break
                
            # print(f"Found potential JPEG at position {jpeg_start}")
            jpeg_end = -1
            for i in range(jpeg_start + 3, len(data) - 1):
                if data[i] == 0xff and data[i+1] == 0xd9:
                    jpeg_end = i + 2
                    break
            
            if jpeg_end > 0:
                jpeg_data = data[jpeg_start:jpeg_end]
                # print(f"Extracted {len(jpeg_data)} bytes of JPEG data")
                
                try:
                    img = tf.image.decode_jpeg(jpeg_data).numpy()
                    if img.shape[0] > 10 and img.shape[1] > 10:
                        jpeg_found = True
                        # plt.figure(figsize=(12, 8))
                        # plt.imshow(img)
                        # plt.title(f"Extracted JPEG at offset {jpeg_start}")
                        # # plt.axis("off")
                        # plt.show()
                        serial.append(img)
                except Exception as e:
                    print(f"Failed to decode JPEG at {jpeg_start}: {e}")
            
            start_pos = jpeg_start + 3
        
        if not jpeg_found:
            print("No valid JPEGs found with direct extraction method")
        images.append(serial)

def count_records_in_tfrecords(file_paths, compression_type=''):
    total_count = 0
    for i, file in enumerate(file_paths):
        dataset = tf.data.TFRecordDataset(file, compression_type=compression_type)
        count = sum(1 for _ in dataset)  # Count lazily
        print(f"[{i+1}/{len(file_paths)}] {file}: {count} records")
        total_count += count
    print(f"\nTotal Records Across {len(file_paths)} Files: {total_count}")
    return total_count



if __name__ == "__main__":

    loader = WaymoDatasetLoader()
    train_files, val_files, test_files = loader.get_file_lists()

    print("Training set size:", len(train_files))
    print("First training file:", train_files[0])

    dataset = tf.data.TFRecordDataset(train_files[:200], compression_type='')

    images = []
    
    # Example usage:
    total = count_records_in_tfrecords(train_files[:5])
    print("Total: ", total)
    
  
