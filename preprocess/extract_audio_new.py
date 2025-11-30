import os
import argparse
import subprocess
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_single_video(args_tuple):
    """
    Worker function to process a single video file.
    Combines ffmpeg and sox via a pipe to avoid intermediate file IO.
    """
    input_f, target_fold = args_tuple
    
    try:
        # --- PRESERVE ORIGINAL FILENAME LOGIC (DO NOT CHANGE) ---
        ext_len = len(input_f.split('/')[-1].split('.')[-1])
        video_id = "-".join(input_f.split('/')[-5:])[:-ext_len-1]
        # --------------------------------------------------------

        output_f = os.path.join(target_fold, video_id + '.wav')

        # set environment vars for sox
        os.environ['LD_LIBRARY_PATH'] = "/home/stud/hunecke/sox/usr/lib/x86_64-linux-gnu" + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['PATH'] = f"/home/stud/hunecke/sox/usr/bin:{os.environ['PATH']}"

        # Construction of the pipeline:
        # ffmpeg (16k resample) -> stdout -> pipe -> stdin -> sox (remix 1) -> file
        
        # 1. ffmpeg command: Output to stdout (-) instead of file
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',               # Overwrite without asking
            '-i', input_f,
            '-vn',              # No video
            '-loglevel', 'error',
            '-ar', '16000',     # Resample 16k
            '-f', 'wav',        # Force wav format for pipe
            '-'                 # Output to stdout
        ]

        # 2. sox command: Input from stdin (-)
        sox_cmd = [
            'sox',
            '-t', 'wav', '-',   # Type wav, input from stdin
            output_f,
            'remix', '1'        # Extract first channel
        ]

        # Execute pipeline
        p1 = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(sox_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Allow p1 to receive a SIGPIPE if p2 exits
        p1.stdout.close()
        
        output, err = p2.communicate()
        
        # Wait for p1 to finish and capture its stderr
        err_ffmpeg = p1.stderr.read()
        p1.wait()
        
        if p1.returncode != 0:
            print(f"FFMPEG Error processing {input_f}: {err_ffmpeg.decode('utf-8', errors='replace').strip()}")
            
        if p2.returncode != 0:
            print(f"SOX Error processing {input_f}: {err.decode('utf-8', errors='replace').strip()}")

    except Exception as e:
        print(f"Error processing {input_f}: {e}")

def main():
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
    # Split the file list based on the Slurm array index
    total_files = len(full_list)
    files_per_shard = np.ceil(total_files / args.num_shards).astype(int)
    start_idx = args.shard_id * files_per_shard
    end_idx = min(start_idx + files_per_shard, total_files)
    
    # Slice the input list
    local_filelist = full_list[start_idx:end_idx]

    print(f"Worker {args.shard_id}/{args.num_shards} processing {len(local_filelist)} files "
          f"(Indices {start_idx} to {end_idx}) using {args.num_workers} threads.")

    # Prepare arguments for map
    # We use a list of tuples to pass multiple args to the worker
    task_args = [(f, args.target_fold) for f in local_filelist]

    # --- PARALLEL EXECUTION ---
    # chunksize=1 ensures better distribution if processing times vary wildy
    with Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_video, task_args, chunksize=1), 
                  total=len(local_filelist), 
                  desc=f'Processing Shard {args.shard_id}'))

if __name__ == "__main__":
    main()