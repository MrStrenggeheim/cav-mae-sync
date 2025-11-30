#%%
import pandas as pd

CSV_PATH = "/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb_1percent.csv"
DATA_PATH = "/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed1pct"

# open csv
csv = pd.read_csv(CSV_PATH)
#%%
csv.head(10)
#%%
data = []

for idx, row in csv.iterrows():
    input_f  = row["video_name"]
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    video_id = "-".join(input_f.split('/')[-5:])[:-ext_len-1]
    item = {
        "video_id": video_id,
        "wav": f"{DATA_PATH}/audio/{video_id}.wav",
        "video_path": f"{DATA_PATH}/frames",
        "labels": row["target"],
    }
    data.append(item)

import json
output = {'data': data}
with open(f"{DATA_PATH}/dataset_info.json", 'w') as f:
    json.dump(output, f, indent=1)
# %%
# also create dumb csv with 
#idx, mid=target, display_name=real/fake
# target_csv = {"index": [0, 1], "mid": ["0", "1"], "display_name": ["real", "fake"]}
# target_df = pd.DataFrame(target_csv)
# target_df.to_csv(f"{DATA_PATH}/dataset_info.csv", index=False)

# %%
