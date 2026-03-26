import tensorflow as tf
from waymo_e2e.data import WaymoDatasetLoader
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

# Map intent integers to strings
INTENT_MAP = {
    1: "GO STRAIGHT",
    2: "GO LEFT",
    3: "GO RIGHT"
}
def count_records_in_tfrecords(file_paths, compression_type=''):
    total_count = 0
    first_left_turn_index = None

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

            if first_left_turn_index is None and intent_val == 2:
                first_left_turn_index = idx

            count += 1
        
        print(f"[{i+1}/{len(file_paths)}] {file}: {count} E2EDFrame instances")
        total_count += count

    # print(f"\nTotal Records Across {len(file_paths)} Files: {total_count}")
    # if first_left_turn_index is not None:
    #     print(f"\nFirst 'go left' (intent=2) appears at record index: {first_left_turn_index}")
    # else:
    #     print("\nNo 'go left' (intent=2) found in the parsed data.")

    return total_count

if __name__ == "__main__":
    loader = WaymoDatasetLoader()
    train_files, val_files, test_files = loader.get_file_lists()

    print("Training set size:", len(train_files))
    print("First training file:", train_files[0])

    total = count_records_in_tfrecords(train_files[:1])  # Count just one file for now
    print("Total: ", total)
