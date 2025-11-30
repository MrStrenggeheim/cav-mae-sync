import argparse
import torch
from torch.utils.data import DataLoader
from src.dataloader_sync import AudiosetDataset, unsupervised_collate_fn
import numpy as np
from tqdm import tqdm
import sys
import os
import logging

def get_args():
    parser = argparse.ArgumentParser(description="Calculate dataset normalization statistics")
    parser.add_argument("--dataset_json", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--target_length", type=int, default=416, help="Target audio length")
    parser.add_argument("--num_mel_bins", type=int, default=128, help="Number of mel bins")
    parser.add_argument("--total_frame", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--im_res", type=int, default=224, help="Image resolution")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()

    audio_conf = {
        'num_mel_bins': args.num_mel_bins,
        'mean': 0, # Placeholder, ignored if skip_norm=True
        'std': 1,  # Placeholder
        'target_length': args.target_length,
        'mode': 'unsupervised_train',
        'total_frame': args.total_frame,
        'im_res': args.im_res,
        'augmentation': False, # No augmentation for stats
        'label_smooth': 0.0,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'custom',
        'skip_norm': True, # Important!
        'noise': False
    }

    logging.info(f"Loading dataset from {args.dataset_json}...")
    dataset = AudiosetDataset(
        dataset_json_file=args.dataset_json,
        audio_conf=audio_conf,
        label_csv=None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=unsupervised_collate_fn,
        pin_memory=True
    )

    logging.info(f"Calculating stats for {len(dataset)} samples...")

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for batch in tqdm(dataloader):
        fbanks, images, video_ids, frame_indices = batch
        # fbanks shape: [B*T, target_length, mel_bins]
        
        fbanks = fbanks.float()
        
        total_sum += fbanks.sum()
        total_sq_sum += (fbanks ** 2).sum()
        total_count += fbanks.numel()

    if total_count == 0:
        logging.error("No data processed. Check dataset and dataloader.")
        return

    mean = total_sum / total_count
    std = (total_sq_sum / total_count - mean ** 2) ** 0.5

    logging.info(f"Calculated Mean: {mean.item()}")
    logging.info(f"Calculated Std: {std.item()}")

if __name__ == "__main__":
    main()
