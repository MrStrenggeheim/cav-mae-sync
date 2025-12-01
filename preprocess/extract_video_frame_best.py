# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

# Critical: Disable internal threading to prevent deadlocks
cv2.setNumThreads(0)

def init_worker():
    cv2.setNumThreads(0)

def resize_and_centercrop(image, target_size=224):
    """
    Standard ResNet-style preprocessing:
    Resize smallest side to 224, then Center Crop 224x224.
    """
    h, w, _ = image.shape
    
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
        
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    c_h, c_w, _ = resized.shape
    start_x = (c_w - target_size) // 2
    start_y = (c_h - target_size) // 2
    
    return resized[start_y:start_y+target_size, start_x:start_x+target_size]

def process_single_video(args_tuple):
    input_video_path, target_fold, extract_frame_num = args_tuple
    
    try:
        if not os.path.exists(input_video_path):
            return

        # --- ID Generation (Matches original) ---
        ext_len = len(input_video_path.split('/')[-1].split('.')[-1])
        video_id = "-".join(input_video_path.split('/')[-5:])[:-ext_len-1]
        
        vidcap = cv2.VideoCapture(input_video_path)
        if not vidcap.isOpened():
            return 
            
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        limit_frames = int(fps * 10)
        eff_total_frames = min(total_frames, limit_frames) if limit_frames > 0 else total_frames
        
        if eff_total_frames <= 0:
            vidcap.release()
            return
            
        indices = [int(i * (eff_total_frames / extract_frame_num)) for i in range(extract_frame_num)]
        
        for i, frame_idx in enumerate(indices):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = vidcap.read()
            
            if ret:
                processed_frame = resize_and_centercrop(frame, 224)
                
                # Path Construction: target_fold/frame_X/video_id.jpg
                save_path = os.path.join(target_fold, f'frame_{i}', f'{video_id}.jpg')
                
                # write BGR image (OpenCV default) to JPG
                cv2.imwrite(save_path, processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        vidcap.release()

    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file_list", type=str, required=True)
    parser.add_argument("-target_fold", type=str, required=True)
    parser.add_argument("--extract_frame_num", type=int, default=16)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    # 1. Create Directories (Locally in Scratch)
    # The structure here (frame_0, frame_1) is preserved during rsync
    for i in range(args.extract_frame_num):
        os.makedirs(os.path.join(args.target_fold, f'frame_{i}'), exist_ok=True)

    # 2. Sharding
    df = pd.read_csv(args.input_file_list)
    col_name = 'video_name' if 'video_name' in df.columns else 'video_path'
    full_list = df[col_name].to_numpy()
    
    total_files = len(full_list)
    files_per_shard = int(np.ceil(total_files / args.num_shards))
    start_idx = args.shard_id * files_per_shard
    end_idx = min(start_idx + files_per_shard, total_files)
    
    local_filelist = full_list[start_idx:end_idx]
    
    print(f"Shard {args.shard_id}: Processing {len(local_filelist)} videos.")

    # 3. Processing
    task_args = [(f, args.target_fold, args.extract_frame_num) for f in local_filelist]

    with Pool(processes=args.num_workers, initializer=init_worker) as pool:
        list(tqdm(pool.imap_unordered(process_single_video, task_args, chunksize=64), 
                  total=len(local_filelist),
                  desc=f"Shard {args.shard_id}"))

if __name__ == "__main__":
    main()