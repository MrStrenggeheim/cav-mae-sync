import argparse
import torch
from torch.utils.data import DataLoader
from src.dataloader_sync import AudiosetDataset
import numpy as np
from tqdm import tqdm
import sys
import os
import logging

def get_args():
    parser = argparse.ArgumentParser(description="Calculate dataset normalization statistics")
    parser.add_argument("--dataset_json", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_mel_bins", type=int, default=128, help="Number of mel bins")
    parser.add_argument("--target_length", type=int, default=1024, help="Target audio length")
    return parser.parse_args()

class AudioOnlyDataset(AudiosetDataset):
    def __getitem__(self, index):
        if index >= self.num_samples:
            return None
        
        datum = self.decode_data(self.data[index])
        try:
            # Extract only audio features
            fbank = self._wav2fbank(datum['wav'])
            return fbank
        except Exception as e:
            logging.warning(f"Error processing {datum['video_id']}: {e}")
            return None

def collate_fn(batch):
    # Filter failed loads
    batch = [b for b in batch if b is not None]
    if not batch:
        logging.warning("All samples in batch failed to load.")
        return torch.empty(0)
    return torch.stack(batch)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()

    audio_conf = {
        'num_mel_bins': args.num_mel_bins,
        'mean': 0, 
        'std': 1,
        'target_length': args.target_length,
        'mode': 'unsupervised_train',
        'total_frame': 1, # unused
        'im_res': 224, # unused
        'augmentation': False,
        'label_smooth': 0.0,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'custom',
        'skip_norm': True,
        'noise': False
    }

    logging.info(f"Loading dataset from {args.dataset_json}...")
    dataset = AudioOnlyDataset(
        dataset_json_file=args.dataset_json,
        audio_conf=audio_conf,
        label_csv=None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    logging.info(f"Calculating stats for {len(dataset)} samples...")

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for fbanks in tqdm(dataloader):
        if fbanks.numel() == 0:
            continue
        
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
