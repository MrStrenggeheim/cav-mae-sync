"""
WebDataset-based data loading pipeline for CAV-MAE Sync.

Replaces ShardedAudiosetDataset with a streaming WebDataset pipeline.
DDP handling is automatic via resampled=True + split_by_node/worker.

Output format is identical to the original pipeline:
  - Per video: (fbanks[16, T, 128], images[16, 3, 224, 224], video_id, frame_indices)
  - After collation: (fbanks[B*16, T, 128], images[B*16, 3, 224, 224], video_ids, frame_indices)
"""

import io
import json
import logging
import os
from glob import glob

import numpy as np
import torch
import torchvision.transforms as T
import webdataset as wds
from PIL import Image


# ---------------------------------------------------------------------------
# Audio slicing (shared logic with dataloader_sharded.py)
# ---------------------------------------------------------------------------

FRAME_SHIFT_MS = 10  # Kaldi fbank frame rate


def slice_fbank_at_timestamp(full_fbank, fbank_length, timestamp_ms, target_length, norm_mean=None, norm_std=None):
    """
    Slice fbank centered at timestamp with edge padding.
    Normalization is applied BEFORE padding so padded zeros represent neutral signal.
    Always returns tensor of shape [target_length, num_mel_bins].
    """
    center_frame = round(timestamp_ms / FRAME_SHIFT_MS)
    half_len = target_length // 2

    start = center_frame - half_len
    end = start + target_length

    actual_length = fbank_length

    pad_left = 0
    pad_right = 0

    if start < 0:
        pad_left = -start
        start = 0
    if end > actual_length:
        pad_right = end - actual_length
        end = actual_length

    segment = full_fbank[start:end, :]

    # Normalize BEFORE padding so padded zeros are neutral
    if norm_mean is not None:
        segment = (segment - norm_mean) / norm_std

    # Pad if needed (zeros are neutral after normalization)
    if pad_left > 0 or pad_right > 0:
        segment = torch.nn.functional.pad(segment, (0, 0, pad_left, pad_right))

    if segment.shape[0] != target_length:
        if segment.shape[0] < target_length:
            pad_needed = target_length - segment.shape[0]
            segment = torch.nn.functional.pad(segment, (0, 0, 0, pad_needed))
        else:
            segment = segment[:target_length, :]

    return segment


# ---------------------------------------------------------------------------
# WebDataset pipeline
# ---------------------------------------------------------------------------

def build_webdataset(
    shard_dir: str,
    audio_conf: dict,
    shuffle: bool = True,
    shuffle_buffer: int = 5000,
    resampled: bool = True,
    samples_per_epoch: int = None,
) -> wds.WebDataset:
    """
    Build a WebDataset pipeline that produces the same output format
    as ShardedAudiosetDataset.flatten_dataset().

    Args:
        shard_dir: Directory containing .tar shard files
        audio_conf: Audio configuration dict (target_length, num_mel_bins, mean, std, etc.)
        shuffle: Whether to shuffle shards and samples
        shuffle_buffer: Size of the cross-shard shuffle buffer
        resampled: Use resampled shards for DDP (infinite stream, handles unequal shards)
        samples_per_epoch: Number of samples per epoch (required when resampled=True).
            Note: with_epoch() operates at the sample level, before DataLoader batching.

    Returns:
        WebDataset pipeline that yields (fbanks, images, video_id, frame_indices) tuples
    """
    # Find tar files
    tar_pattern = os.path.join(shard_dir, "*.tar")
    tar_files = sorted(glob(tar_pattern))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {shard_dir}")
    logging.info(f"Found {len(tar_files)} shards in {shard_dir}")

    # Extract config
    target_length = audio_conf.get('target_length', 48)
    num_mel_bins = audio_conf.get('num_mel_bins', 128)
    skip_norm = audio_conf.get('skip_norm', False)
    norm_mean = audio_conf.get('mean', None) if not skip_norm else None
    norm_std = audio_conf.get('std', None) if not skip_norm else None
    im_res = audio_conf.get('im_res', 224)
    total_frame = audio_conf.get('total_frame', 16)
    augmentation = audio_conf.get('augmentation', False)

    # Image transforms (same as ShardedAudiosetDataset)
    imagenet_mean = [0.4850, 0.4560, 0.4060]
    imagenet_std = [0.2290, 0.2240, 0.2250]
    normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)

    if augmentation:
        image_transform = T.Compose([
            T.RandomResizedCrop(im_res, scale=(0.08, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize,
        ])
    else:
        image_transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])

    def decode_and_preprocess(sample):
        """Decode a WebDataset sample → (fbanks, images, video_id, frame_indices) tuple."""
        # Parse metadata
        meta = json.loads(sample['json'].decode('utf-8') if isinstance(sample['json'], bytes) else sample['json'])
        video_id = meta['video_id']
        fbank_length = meta['fbank_length']
        frame_timestamps_ms = meta['frame_timestamps_ms']
        frame_indices_list = meta['frame_indices']

        # Load fbank
        fbank_bytes = sample['fbank.npy']
        full_fbank = torch.from_numpy(
            np.load(io.BytesIO(fbank_bytes) if isinstance(fbank_bytes, bytes) else fbank_bytes)
        ).float()

        # Process each frame
        fbanks = []
        images = []

        for i in range(min(total_frame, len(frame_timestamps_ms))):
            # Audio: slice at timestamp with normalize-before-pad
            timestamp_ms = frame_timestamps_ms[i]
            fbank = slice_fbank_at_timestamp(
                full_fbank, fbank_length, timestamp_ms, target_length,
                norm_mean=norm_mean, norm_std=norm_std,
            )

            # Defensive shape check
            if fbank.shape != (target_length, num_mel_bins):
                t, f = fbank.shape
                if f != num_mel_bins:
                    fbank = fbank[:, :num_mel_bins] if f > num_mel_bins else torch.nn.functional.pad(fbank, (0, num_mel_bins - f))
                if t != target_length:
                    fbank = fbank[:target_length, :] if t > target_length else torch.nn.functional.pad(fbank, (0, 0, 0, target_length - t))

            fbanks.append(fbank)

            # Image: decode JPEG and apply transforms
            frame_key = f'{i:03d}.jpg'
            img_bytes = sample[frame_key]
            img = Image.open(io.BytesIO(img_bytes) if isinstance(img_bytes, bytes) else img_bytes).convert('RGB')
            img_tensor = image_transform(img)
            images.append(img_tensor)

        fbanks = torch.stack(fbanks)       # [16, target_length, 128]
        images = torch.stack(images)       # [16, 3, 224, 224]
        frame_indices = torch.tensor(frame_indices_list[:total_frame], dtype=torch.long)

        return fbanks, images, video_id, frame_indices

    # Build pipeline
    dataset = wds.WebDataset(
        tar_files,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.shardlists.split_by_worker,
        resampled=resampled,
        shardshuffle=shuffle,
    )

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)

    # Handle corrupt shard entries gracefully (log warning, skip entry)
    dataset = dataset.map(decode_and_preprocess, handler=wds.warn_and_continue)

    if resampled and samples_per_epoch is not None:
        # with_epoch() operates at sample level (before DataLoader batching)
        dataset = dataset.with_epoch(samples_per_epoch)

    return dataset


def unsupervised_collate_fn(batch):
    """
    Collate function for unsupervised training.
    Identical to dataloader_sync.unsupervised_collate_fn.
    """
    fbanks, images, video_ids, frame_indices = zip(*batch)

    fbanks = torch.cat(fbanks)     # [B*16, T, 128]
    images = torch.cat(images)     # [B*16, 3, 224, 224]

    expanded_video_ids = []
    for i, vid in enumerate(video_ids):
        num_frames = frame_indices[i].shape[0]
        expanded_video_ids.extend([vid] * num_frames)

    frame_indices = torch.cat(frame_indices)

    return fbanks, images, expanded_video_ids, frame_indices
