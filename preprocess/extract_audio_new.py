import os
import argparse
import subprocess
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def get_video_id(input_f):
    """
    Extracts video ID from filename using specific logic.
    Refactored for cleanliness, matches Version 1 logic.
    """
    try:
        parts = input_f.split('/')
        filename = parts[-1]
        ext_len = len(filename.split('.')[-1])
        # Join last 5 parts, remove extension (logic from V1)
        return "-".join(parts[-5:])[:-ext_len-1]
    except Exception:
        # Fallback if path is shorter than expected
        return os.path.splitext(os.path.basename(input_f))[0]

def process_single_video(args_tuple):
    """
    Worker function to process a single video file.
    Uses ffmpeg directly to extract audio (16k, mono, channel 1).
    """
    input_f, target_fold = args_tuple
    
    try:
        video_id = get_video_id(input_f)
        output_f = os.path.join(target_fold, video_id + '.wav')

        # Skip if exists to allow resume (Logic from V1)
        # We check here so TQDM shows progress even for existing files
        if os.path.exists(output_f):
            return

        # FFmpeg command equivalent to: ffmpeg (16k) -> sox (remix 1)
        # -af "pan=mono|c0=c0" selects the first channel (c0) as the mono output
        cmd = [
            'ffmpeg',
            '-y',               # Overwrite
            '-v', 'error',      # Log level error
            '-i', input_f,
            '-vn',              # No video
            '-ar', '16000',     # Resample 16k
            '-af', 'pan=mono|c0=c0', # Remix 1 equivalent
            '-f', 'wav',
            '-threads', '1',    # Crucial for multiprocessing
            output_f
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    except subprocess.CalledProcessError as e:
        # Tqdm-safe printing
        tqdm.write(f"FFmpeg error for {input_f}: {e.stderr.decode().strip()}")
    except Exception as e:
        tqdm.write(f"Error processing {input_f}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Optimized video feature extractor')
    parser.add_argument("-input_file_list", type=str, required=True, help="Path to CSV file.")
    parser.add_argument("-target_fold", type=str, default='./sample_audio/', help="Output directory.")
    parser.add_argument("--shard_id", type=int, default=0, help="Slurm array task ID.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total Slurm array tasks.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of local processes.")
    args = parser.parse_args()

    # Load Data
    df = pd.read_csv(args.input_file_list)
    full_list = df['video_name'].to_numpy()

    # Create directory
    os.makedirs(args.target_fold, exist_ok=True)

    # --- SHARDING LOGIC ---
    total_files = len(full_list)
    files_per_shard = np.ceil(total_files / args.num_shards).astype(int)
    start_idx = args.shard_id * files_per_shard
    end_idx = min(start_idx + files_per_shard, total_files)
    
    local_filelist = full_list[start_idx:end_idx]

    print(f"Worker {args.shard_id}/{args.num_shards} processing {len(local_filelist)} files "
          f"(Indices {start_idx} to {end_idx}) using {args.num_workers} threads.")

    # Prepare arguments
    task_args = [(f, args.target_fold) for f in local_filelist]

    # --- PARALLEL EXECUTION ---
    # chunksize=1 ensures the progress bar updates smoothly
    with Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_video, task_args, chunksize=1), 
                  total=len(local_filelist), 
                  desc=f'Shard {args.shard_id}'))

if __name__ == "__main__":
    main()