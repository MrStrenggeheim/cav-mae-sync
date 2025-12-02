#%%
import pandas as pd
import json
import os

DATA_PATH = "/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed"
CSV_PATHS = [
    # "/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb.csv",
    # "/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb_1percent.csv",
    # "/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb_test.csv",
    # "/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb_train.csv"
    '/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb_10percent_train.csv',
    '/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb_10percent_test.csv'
]


def video_name_to_video_id(input_f):
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    video_id = "-".join(input_f.split('/')[-5:])[:-ext_len-1]
    return video_id

def process_csv(csv_path):
    print(f"Processing CSV: {csv_path}")
    csv = pd.read_csv(csv_path)
    data = []

    for idx, row in csv.iterrows():
        input_f  = row["video_name"]
        video_id = video_name_to_video_id(input_f)
        item = {
            "video_id": video_id,
            "wav": f"{DATA_PATH}/audio/{video_id}.wav",
            "video_path": f"{DATA_PATH}/frames",
            "labels": row["target"],
        }
        data.append(item)

    output = {'data': data}
    json_file_name = os.path.basename(csv_path).replace('.csv', '_dataset_info.json')
    with open(f"{DATA_PATH}/{json_file_name}", 'w') as f:
        json.dump(output, f, indent=1)

for csv_path in CSV_PATHS:
    process_csv(csv_path)