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

def plot_8_images_side_by_side(image_list, titles=None, figsize=(24, 4)):
        """
        Plots 8 images in a 1x8 subplot layout.

        Args:
            image_list (list or np.ndarray): List of 8 images (as NumPy arrays).
            titles (list of str, optional): Titles for each subplot.
            figsize (tuple): Figure size for the entire plot.
        """
        if len(image_list) != 8:
            raise ValueError("Exactly 8 images are required.")

        plt.figure(figsize=figsize)
        
        for i, img in enumerate(image_list):
            plt.subplot(1, 8, i + 1)
            plt.imshow(img)
            plt.axis('off')
            if titles and i < len(titles):
                plt.title(titles[i], fontsize=10)

        plt.tight_layout()
        plt.show()

def plot_front_3_camera_images(left_image, front_image, right_image, title="Front 3 Camera Views"):
        """
        Plots the front_left, front, and front_right camera images side by side.

        Args:
            left_image (np.ndarray): Image from the front_left camera.
            front_image (np.ndarray): Image from the front camera.
            right_image (np.ndarray): Image from the front_right camera.
            title (str): Title for the plot (optional).
        """
        image_list = [left_image, front_image, right_image]
        
        # Check all images have the same height
        heights = [img.shape[0] for img in image_list]
        if len(set(heights)) != 1:
            raise ValueError("All images must have the same height to concatenate horizontally.")
        
        concatenated_image = np.concatenate(image_list, axis=1)

        plt.figure(figsize=(20, 10))
        plt.imshow(concatenated_image)
        plt.axis('off')
        plt.title(title)
        plt.show()

if __name__ == "__main__":

    loader = WaymoDatasetLoader()
    train_files, val_files, test_files = loader.get_file_lists()

    print("Training set size:", len(train_files))
    print("First training file:", train_files[0])

    matrix = []
    for train_file in train_files[:2]:
        gcs_path = train_file
        print("Attempting to extract images from Waymo End-to-End Camera dataset...")
        print(f"Using path: {gcs_path}")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        try:
            images = []
            extract_image_from_waymo_endtoend(gcs_path)
            # print("\n\nTrying direct image extraction method...")
            try_extract_common_formats(gcs_path)
            # print(len(images))

            # Plotting all 8 images in a row!
            # plot_8_images_side_by_side(images[0])
            img_np = np.array(images, dtype=object)

            # for image in img_np:
            #     print(image.shape)

            # Front view!
            # plot_front_3_camera_images(img_np[0][0], img_np[0][1], img_np[0][2])
            
            print(img_np.shape)
            matrix.append(img_np)
            # Save for later!
            # np.save("numpy_data/first_row.npy", img_np)
            
        except Exception as e:
            print(f"Failed: {e}")
            print("1. Ensure you have the right access credentials for the Waymo dataset")
            print("2. This script is designed specifically for the End-to-End Camera dataset")
            print("3. You might need to download the file locally first:")
            print("   - gsutil cp gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training_202504031202_202504151040.tfrecord-00000-of-00263 ./")
        
    # matrix = np.stack(matrix, axis = 0)
    # print(matrix.shape)
    np.save("numpy_data/first_two.npy", matrix)
