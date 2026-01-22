#!/usr/bin/env python3
"""
Generate or update shard metadata for existing shards.

This script counts samples in each shard and saves the counts to sharded_dataset.json.
Run this ONCE after creating shards, or if you need to update metadata for existing shards.

Usage:
    python scripts/generate_shard_metadata.py --shard_dir /path/to/shards

This avoids the need to load all shards into memory during training startup.
"""

import argparse
import os
import json
import glob
import torch
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def count_shard_samples(shard_path):
    """Count samples in a single shard without keeping data in memory."""
    try:
        data_list = torch.load(shard_path, weights_only=False)
        count = len(data_list)
        del data_list  # Free memory immediately
        return count
    except Exception as e:
        logging.warning(f"Error loading {shard_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Generate shard metadata for efficient DDP sync")
    parser.add_argument("--shard_dir", type=str, required=True, help="Directory containing .pt shards")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output metadata file (default: <shard_dir>/sharded_dataset.json)")
    args = parser.parse_args()

    shard_files = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.pt")))
    if not shard_files:
        logging.error(f"No shard files found in {args.shard_dir}")
        return 1
    
    logging.info(f"Found {len(shard_files)} shards in {args.shard_dir}")
    
    # Count samples in each shard
    shard_sample_counts = {}
    total_samples = 0
    
    for shard_path in tqdm(shard_files, desc="Counting samples"):
        shard_name = os.path.basename(shard_path)
        count = count_shard_samples(shard_path)
        shard_sample_counts[shard_name] = count
        total_samples += count
    
    # Load existing metadata if present
    output_path = args.output or os.path.join(args.shard_dir, "sharded_dataset.json")
    existing_metadata = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_metadata = json.load(f)
            logging.info(f"Loaded existing metadata from {output_path}")
        except Exception as e:
            logging.warning(f"Could not load existing metadata: {e}")
    
    # Update metadata
    existing_metadata['shard_sample_counts'] = shard_sample_counts
    existing_metadata['total_samples'] = total_samples
    existing_metadata['num_shards'] = len(shard_files)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(existing_metadata, f, indent=2)
    
    logging.info(f"Saved metadata to {output_path}")
    logging.info(f"Total: {total_samples} samples across {len(shard_files)} shards")
    logging.info(f"Average: {total_samples / len(shard_files):.1f} samples per shard")
    
    return 0


if __name__ == "__main__":
    exit(main())
