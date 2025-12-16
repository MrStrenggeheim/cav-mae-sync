import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import glob
import os

def get_args():
    parser = argparse.ArgumentParser(description="Calculate dataset normalization statistics")
    parser.add_argument("--shard_dir", type=str, required=True, help="Path to sharded dataset directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max_shards", type=int, default=None, help="Max shards to process (for faster estimation)")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()

    shards = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.pt")))
    if not shards:
        logging.error(f"No shards found in {args.shard_dir}")
        return
    
    if args.max_shards:
        shards = shards[:args.max_shards]
    
    logging.info(f"Processing {len(shards)} shards from {args.shard_dir}...")

    # Welford's online algorithm for numerically stable mean/variance
    count = 0
    mean = 0.0
    M2 = 0.0  # Sum of squared differences from current mean

    for shard_path in tqdm(shards, desc="Processing shards"):
        try:
            data_list = torch.load(shard_path, weights_only=False, mmap=True)
            
            for item in data_list:
                fbank = item['fbank'].float()
                fbank_length = item.get('fbank_length', fbank.shape[0])
                fbank = fbank[:fbank_length].flatten()
                
                # Batch update using Welford's parallel algorithm
                batch_count = fbank.numel()
                batch_mean = fbank.mean().item()
                batch_var = fbank.var().item() if batch_count > 1 else 0.0
                batch_M2 = batch_var * batch_count
                
                # Combine with running stats
                delta = batch_mean - mean
                new_count = count + batch_count
                mean = mean + delta * batch_count / new_count
                M2 = M2 + batch_M2 + delta ** 2 * count * batch_count / new_count
                count = new_count
                
        except Exception as e:
            logging.warning(f"Error processing {shard_path}: {e}")
            continue

    if count == 0:
        logging.error("No data processed.")
        return

    variance = M2 / count
    std = variance ** 0.5

    logging.info(f"Processed {count:,} values")
    logging.info(f"Mean: {mean:.12f}")
    logging.info(f"Std:  {std:.12f}")
    print()
    print("# Add to FakeSyncConfig or audio_conf:")
    print(f"mean: float = {mean}")
    print(f"std: float = {std}")

if __name__ == "__main__":
    main()

