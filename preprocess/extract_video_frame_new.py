# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import argparse
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
from multiprocessing import Pool

# Global transform definition (stateless, safe to share)
preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()])

def init_worker():
    """
    Initializer to prevent thread contention. 
    OpenCV and PyTorch try to be smart with threading, which freezes Multiprocessing pools.
    """
    cv2.setNumThreads(0)
    torch.set_num_threads(1)

def process_single_video(args_tuple):
    """
    Worker function to process a single video.
    """
    input_video_path, target_fold, extract_frame_num = args_tuple
    
    try:
        # --- PRESERVE ORIGINAL LOGIC (Filename parsing) ---
        ext_len = len(input_video_path.split('/')[-1].split('.')[-1])
        video_id = "-".join(input_video_path.split('/')[-5:])[:-ext_len-1]
        
        # Open Video
        vidcap = cv2.VideoCapture(input_video_path)
        if not vidcap.isOpened():
            return # Skip broken videos silently
            
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        
        # --- PRESERVE ORIGINAL LOGIC (Frame counting) ---
        total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))
        
        for i in range(extract_frame_num):
            frame_idx = int(i * (total_frame_num/extract_frame_num))
            
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = vidcap.read()
            
            if ret:
                cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                image_tensor = preprocess(pil_im)
                
                # Save path (Folders are pre-created in main)
                save_path = os.path.join(target_fold, 'frame_{:d}'.format(i), video_id + '.jpg')
                save_image(image_tensor, save_path)
        
        vidcap.release()

    except Exception as e:
        print(f"Error processing {input_video_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Optimized video frame extractor")
    parser.add_argument("-input_file_list", type=str, required=True)
    parser.add_argument("-target_fold", type=str, default='./sample_frames/')
    parser.add_argument("--extract_frame_num", type=int, default=16, help="Number of frames to extract per video")
    
    # Slurm/Parallel args
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # 1. Prepare Output Directories ONCE (Performance Critical)
    # We create frame_0, frame_1, ... frame_N immediately.
    print(f"Ensuring output directories exist for {args.extract_frame_num} frames...")
    for i in range(args.extract_frame_num):
        dir_path = os.path.join(args.target_fold, 'frame_{:d}'.format(i))
        os.makedirs(dir_path, exist_ok=True)

    # 2. Load and Shard Data
    df = pd.read_csv(args.input_file_list)
    
    # Handle column name (fallback to video_path if video_name missing)
    col_name = 'video_name' if 'video_name' in df.columns else 'video_path'
    full_list = df[col_name].to_numpy()
    
    total_files = len(full_list)
    files_per_shard = int(np.ceil(total_files / args.num_shards))
    start_idx = args.shard_id * files_per_shard
    end_idx = min(start_idx + files_per_shard, total_files)
    
    local_filelist = full_list[start_idx:end_idx]
    
    print(f"Shard {args.shard_id}/{args.num_shards}: Processing {len(local_filelist)} videos.")

    # 3. Prepare Worker Arguments
    # pass args.extract_frame_num dynamically
    task_args = [(f, args.target_fold, args.extract_frame_num) for f in local_filelist]

    # 4. Run Parallel Processing
    with Pool(processes=args.num_workers, initializer=init_worker) as pool:
        list(tqdm(pool.imap_unordered(process_single_video, task_args, chunksize=2), 
                  total=len(local_filelist),
                  desc=f"Shard {args.shard_id}"))

if __name__ == "__main__":
    main()