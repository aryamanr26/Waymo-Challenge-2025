import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset

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