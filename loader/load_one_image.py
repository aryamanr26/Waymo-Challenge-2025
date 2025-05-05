import tensorflow as tf
import matplotlib.pyplot as plt
from waymo_open_dataset import dataset_pb2 as open_dataset

# Direct GCS path string — no gcsfs
gcs_path = 'gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training_202504031202_202504151040.tfrecord-00000-of-00263'

# Load the first record from the TFRecord file
dataset = tf.data.TFRecordDataset(gcs_path, compression_type='')

for data in dataset.take(1):  # just one frame
    frame = open_dataset.Frame()
    frame.ParseFromString(data.numpy())

    # Visualize the front camera
    for image in frame.images:
        if image.name == open_dataset.CameraName.FRONT:
            img = tf.image.decode_jpeg(image.image).numpy()
            plt.imshow(img)
            plt.title("Front Camera")
            plt.axis("off")
            plt.show()
            break
