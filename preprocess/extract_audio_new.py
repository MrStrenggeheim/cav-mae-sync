import os
import argparse
import subprocess
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

def get_video_id(input_f):
    """
    Extracts video ID from filename using the specific logic.
    """
    parts = input_f.split('/')
    filename = parts[-1]
    ext_len = len(filename.split('.')[-1])
    # Join last 5 parts, remove extension
    return "-".join(parts[-5:])[:-ext_len-1]

def process_single_video(args_tuple):
    """
    Worker function to process a single video file.
    Uses ffmpeg directly to extract audio and select first channel.
    """
    input_f, target_fold = args_tuple
    
    try:
        video_id = get_video_id(input_f)
        output_f = os.path.join(target_fold, video_id + '.wav')

        cmd = [
            'ffmpeg',
            '-y',
            '-v', 'error',
            '-i', input_f,
            '-vn',
            '-ar', '16000',
            '-af', 'pan=mono|c0=c0',
            '-threads', '1',
            output_f
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    except subprocess.CalledProcessError as e:
        logging.warning(f"FFmpeg error for {input_f}: {e.stderr.decode().strip()}")
        # Do not raise, just log and skip to allow other files to process
    except Exception as e:
        logging.warning(f"Error processing {input_f}: {e}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Optimized video feature extractor')
    parser.add_argument("-input_file_list", type=str, required=True, help="Path to CSV file.")
    parser.add_argument("-target_fold", type=str, default='./sample_audio/', help="Output directory.")
    # Slurm Array Support
    parser.add_argument("--shard_id", type=int, default=0, help="Current Slurm array task ID (0-indexed).")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of Slurm array tasks.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of local parallel processes.")
    args = parser.parse_args()

    # Load Data
    df = pd.read_csv(args.input_file_list)
    full_list = df['video_name'].to_numpy()

    # Create directory (Race condition safe)
    os.makedirs(args.target_fold, exist_ok=True)

    # --- SHARDING LOGIC ---
    total_files = len(full_list)
    files_per_shard = np.ceil(total_files / args.num_shards).astype(int)
    start_idx = args.shard_id * files_per_shard
    end_idx = min(start_idx + files_per_shard, total_files)
    
    local_filelist = full_list[start_idx:end_idx]

    logging.info(f"Checking existing files in {args.target_fold}...")
    try:
        existing_files = set(os.listdir(args.target_fold))
    except FileNotFoundError as e:
        existing_files = set()
        logging.error(f"Error accessing target folder: {e}")
        raise e
        
    # Filter list
    files_to_process = []
    for f in local_filelist:
        vid = get_video_id(f)
        if f"{vid}.wav" not in existing_files:
            files_to_process.append(f)
            
    logging.info(f"Worker {args.shard_id}/{args.num_shards}: {len(files_to_process)}/{len(local_filelist)} files to process "
          f"(Indices {start_idx} to {end_idx}) using {args.num_workers} threads.")

    if not files_to_process:
        logging.warning("No files to process.")
        return

    # Prepare arguments for map
    task_args = [(f, args.target_fold) for f in files_to_process]

    chunk_size = max(1, len(files_to_process) // (args.num_workers * 4))
    
    with Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_video, task_args, chunksize=chunk_size), 
                  total=len(files_to_process), 
                  desc=f'Processing Shard {args.shard_id}'))

if __name__ == "__main__":
    main()