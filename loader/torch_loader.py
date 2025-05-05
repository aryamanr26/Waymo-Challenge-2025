import torch
from torch.utils.data import IterableDataset, DataLoader
import tensorflow as tf  # Only used for TFRecord parsing
from waymo_dataset_loader import WaymoDatasetLoader
import io
import numpy as np
from PIL import Image

class WaymoTorchDataset(IterableDataset):
    def __init__(self, file_patterns, shuffle_files=False):
        self.file_patterns = file_patterns
        self.shuffle_files = shuffle_files
        
        # Get actual file paths
        # self.loader = WaymoDatasetLoader()
        # self.train_files, self.val_files, self.test_files = self.loader.get_file_lists()
        self.files = self.file_patterns # Modify based on your needs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # Shard files across workers
        sharded_files = self.files[worker_id::num_workers]
        
        if self.shuffle_files:
            np.random.shuffle(sharded_files)

        for file_path in sharded_files:
            dataset = tf.data.TFRecordDataset(file_path, compression_type='')
            
            for raw_record in dataset:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                # Extract images
                image_tensors = []
                for key in example.features.feature:
                    feature = example.features.feature[key]
                    if feature.HasField('bytes_list') and feature.bytes_list.value:
                        byte_data = feature.bytes_list.value[0]
                        
                        # Check for JPEG header
                        if byte_data.startswith(b'\xff\xd8\xff'):
                            try:
                                # Convert to PIL Image then to tensor
                                img = Image.open(io.BytesIO(byte_data))
                                img_tensor = torch.from_numpy(np.array(img))
                                image_tensors.append(img_tensor)
                            except Exception as e:
                                print(f"Error decoding image: {e}")
                                continue
                                
                if image_tensors:
                    yield torch.stack(image_tensors)  # Stack images from single frame

    @staticmethod
    def collate_fn(batch):
        # Custom collation if needed
        return batch

# Usage example:
if __name__ == "__main__":
    loader = WaymoDatasetLoader()
    train_files, val_files, test_files = loader.get_file_lists()
    
    # Use a subset for testing, e.g., the first file
    dataset = WaymoTorchDataset(train_files[:1])  
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,           # Start small for testing
        num_workers=4,          # Set >0 for parallel loading
        collate_fn=WaymoTorchDataset.collate_fn,
    )
    
    for batch in dataloader:
        print(f"Batch size: {len(batch)}")
        print(f"Shape of first item: {batch[0].shape}")