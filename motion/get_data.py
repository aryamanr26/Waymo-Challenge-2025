import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # (optional) disables oneDNN custom ops warning

import tensorflow as tf
import matplotlib.pyplot as plt
import gcsfs
import matplotlib.animation as animation
from helper import visualize_all_agents_smooth, features_description

def create_animation(images):
    print("Creating animation with", len(images), "frames.")
    fig, ax = plt.subplots()
    dpi = 100
    size_inches = 1000 / dpi
    fig.set_size_inches([size_inches, size_inches])

    def animate_func(i):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"Frame {i + 1}/{len(images)}")

    anim = animation.FuncAnimation(
        fig, animate_func, frames=len(images), interval=100)
    return anim

class WaymoMotionDatasetCounter:
    def __init__(self, bucket_name="waymo_open_dataset_motion_v_1_3_0", gcs_prefix="uncompressed/tf_example/training"):
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.fs = gcsfs.GCSFileSystem(token='google_default')

    def count_files(self):
        # Compose the full GCS path
        gcs_path = f"{self.bucket_name}/{self.gcs_prefix}/*"
        print(f"Counting files in: gs://{self.bucket_name}/{self.gcs_prefix}/")
        files = self.fs.glob(gcs_path)
        print(f"Number of files in GCS folder: {len(files)}")
        return len(files)

counter = WaymoMotionDatasetCounter()
counter.count_files()

# Waymo state_features description:
# - 128 agents per scene.
# - Each agent has past (10), current (1), and future (80) timesteps.

# Common attributes (per agent):
# - 'state/id': unique object ID.
# - 'state/type': object type (car, pedestrian, etc.).
# - 'state/is_sdc': whether it's the self-driving car (1 if true).
# - 'state/tracks_to_predict': whether this agent's future is to be predicted.

# Temporal features:
# - 'past', 'current', and 'future' subgroups contain:
#   - Position: x, y, z
#   - Velocity: velocity_x, velocity_y, vel_yaw
#   - Box: bbox_yaw (heading), length, width, height
#   - Timestamp: timestamp_micros
#   - Validity: valid (1 = valid, 0 = missing/occluded)

# Shapes:
# - past: [128, 10]
# - current: [128, 1]
# - future: [128, 80]

# Direct GCS path string — no gcsfs
gcs_path = 'gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000'

# Load the first record from the TFRecord file
dataset = tf.data.TFRecordDataset(gcs_path, compression_type='')

# num_samples = sum(1 for _ in dataset)
# print("Number of samples in dataset:", num_samples)

image_stack = []
for data in dataset.take(1):  # just one frame
    parsed = tf.io.parse_single_example(data, features_description)
    #parsed['state/is_sdc'] = tf.cast(parsed['state/is_sdc'], tf.int64)  # Ensure is_sdc is int64
    #print("state/is_sdc array", parsed['state/is_sdc']) 
    images = visualize_all_agents_smooth(parsed)
    print("Generated images for all agents.", len(images), "images created.")
    print("First image shape:", images[0].shape)
    image_stack.append(images)

combined = [item for sublist in image_stack for item in sublist]
anim = create_animation(combined[::5]) # Show every 5th frame for speed
plt.show()



