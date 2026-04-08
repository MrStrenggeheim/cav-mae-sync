#!/usr/bin/env python3
"""
Create WebDataset .tar shards from video files with identity-aware splitting.

Each sample in the tar shard contains:
  {key}.json      - metadata (video_id, fbank_length, timestamps, fps, labels, identity_id)
  {key}.fbank.npy - mel-spectrogram [T, 128] float16
  {key}.000.jpg   - Frame 0 (224x224, JPEG q=90)
  ...
  {key}.015.jpg   - Frame 15

Usage:
  # VoxCeleb2 with identity-aware split (exclude FakeAVCeleb identities)
  python create_webdataset.py \\
      --input_file voxceleb2_train.csv \\
      --output_dir ./shards/ \\
      --split_by_identity --split_ratios 0.9 0.05 0.05 \\
      --exclude_identities_file fakeavceleb_identities.txt

  # FakeAVCeleb (pre-split CSV, no further splitting)
  python create_webdataset.py \\
      --input_file fakeavceleb_eval.csv \\
      --output_dir ./shards_eval/ \\
      --split_name test
"""

import argparse
import io
import json
import logging
import os
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio.compliance.kaldi as kaldi
import torchvision.transforms as T
import webdataset as wds
from PIL import Image
from tqdm import tqdm

try:
    from decord import VideoReader, cpu as decord_cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    logging.warning("decord not available, falling back to OpenCV for video decoding")
    import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Identity extraction
# ---------------------------------------------------------------------------

def extract_identity(video_path: str) -> str:
    """Extract identity ID from video path (e.g., 'id00001' from VoxCeleb2 paths)."""
    match = re.search(r'(id\d+)', video_path)
    if match:
        return match.group(1)
    # Fallback: use parent directory name
    return Path(video_path).parent.name


def create_identity_splits(
    file_list: list,
    split_ratios: tuple = (0.9, 0.05, 0.05),
    exclude_identities: set = None,
    seed: int = 42,
) -> dict:
    """
    Split file_list by identity into train/val/test sets.
    Ensures no identity overlap between splits.

    Returns dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    # Extract identities
    for item in file_list:
        item['identity'] = extract_identity(item['video_path'])

    # Exclude specified identities (e.g., FakeAVCeleb identities from VoxCeleb2)
    if exclude_identities:
        before = len(file_list)
        file_list = [f for f in file_list if f['identity'] not in exclude_identities]
        excluded = before - len(file_list)
        logging.info(f"Excluded {excluded} videos from {len(exclude_identities)} identities")

    # Group by identity
    identity_to_videos = {}
    for item in file_list:
        identity_to_videos.setdefault(item['identity'], []).append(item)

    identities = sorted(identity_to_videos.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(identities)

    n = len(identities)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train:n_train + n_val])
    test_ids = set(identities[n_train + n_val:])

    splits = {'train': [], 'val': [], 'test': []}
    for identity, videos in identity_to_videos.items():
        if identity in train_ids:
            splits['train'].extend(videos)
        elif identity in val_ids:
            splits['val'].extend(videos)
        else:
            splits['test'].extend(videos)

    logging.info(
        f"Identity split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test identities, "
        f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])} videos"
    )
    return splits


# ---------------------------------------------------------------------------
# Video/audio processing (preserves exact semantics from create_sharded_dataset.py)
# ---------------------------------------------------------------------------

def extract_audio_fbank(video_path: str, num_mel_bins: int = 128, max_audio_length: int = 1024):
    """Extract mel-spectrogram from video using ffmpeg + Kaldi fbank."""
    command = [
        'ffmpeg', '-v', 'quiet', '-i', video_path,
        '-f', 's16le', '-ar', '16000', '-ac', '1',
        '-af', 'pan=mono|c0=c0', '-'
    ]
    pipe = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    raw_audio = np.frombuffer(pipe.stdout, dtype=np.int16).copy()
    waveform = torch.from_numpy(raw_audio).float().unsqueeze(0) / 32768.0
    sr = 16000

    waveform = waveform - waveform.mean()
    full_fbank = kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr,
        use_energy=False, window_type='hanning',
        num_mel_bins=num_mel_bins, dither=0.0, frame_shift=10
    )

    n_frames = full_fbank.shape[0]
    fbank_length = min(n_frames, max_audio_length)

    if n_frames > max_audio_length:
        full_fbank = full_fbank[:max_audio_length, :]
    elif n_frames < max_audio_length:
        pad_len = max_audio_length - n_frames
        full_fbank = torch.nn.functional.pad(full_fbank, (0, 0, 0, pad_len))

    return full_fbank.half(), fbank_length


def extract_frames_decord(video_path: str, num_frames: int, im_res: int, max_audio_length: int):
    """Extract frames using decord VideoReader with timestamp-based seeking."""
    vr = VideoReader(video_path, ctx=decord_cpu(0))
    fps = vr.get_avg_fps()
    total_video_frames = len(vr)

    max_video_duration_ms = max_audio_length * 10
    max_video_frames = int((max_video_duration_ms / 1000) * fps) if fps > 0 else total_video_frames
    usable_video_frames = min(total_video_frames, max_video_frames)
    video_duration_ms = (usable_video_frames / fps) * 1000 if fps > 0 else 0

    pil_transform = T.Compose([T.Resize(im_res), T.CenterCrop(im_res)])

    images_bytes = []
    frame_indices = []
    frame_timestamps_ms = []

    for i in range(num_frames):
        frame_idx = int(i * (usable_video_frames / num_frames)) if usable_video_frames > 0 else 0
        timestamp_ms = (frame_idx / fps) * 1000 if fps > 0 else 0

        try:
            frame = vr[frame_idx].asnumpy()  # RGB numpy array [H, W, C]
            pil_im = Image.fromarray(frame)
            pil_im = pil_transform(pil_im)
        except Exception as e:
            logging.warning(f"Frame {frame_idx} failed for {video_path}: {e}, using black frame")
            pil_im = Image.new('RGB', (im_res, im_res))

        img_byte_arr = io.BytesIO()
        pil_im.save(img_byte_arr, format='JPEG', quality=90)
        images_bytes.append(img_byte_arr.getvalue())
        frame_indices.append(frame_idx)
        frame_timestamps_ms.append(timestamp_ms)

    return images_bytes, frame_indices, frame_timestamps_ms, fps, video_duration_ms


def extract_frames_opencv(video_path: str, num_frames: int, im_res: int, max_audio_length: int):
    """Extract frames using OpenCV (fallback when decord unavailable)."""
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_video_duration_ms = max_audio_length * 10
    max_video_frames = int((max_video_duration_ms / 1000) * fps) if fps > 0 else total_video_frames
    usable_video_frames = min(total_video_frames, max_video_frames)
    video_duration_ms = (usable_video_frames / fps) * 1000 if fps > 0 else 0

    pil_transform = T.Compose([T.Resize(im_res), T.CenterCrop(im_res)])

    images_bytes = []
    frame_indices = []
    frame_timestamps_ms = []

    for i in range(num_frames):
        frame_idx = int(i * (usable_video_frames / num_frames)) if usable_video_frames > 0 else 0
        timestamp_ms = (frame_idx / fps) * 1000 if fps > 0 else 0

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = vidcap.read()

        if ret:
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            pil_im = pil_transform(pil_im)
        else:
            logging.warning(f"Frame {frame_idx} failed for {video_path}, using black frame")
            pil_im = Image.new('RGB', (im_res, im_res))

        img_byte_arr = io.BytesIO()
        pil_im.save(img_byte_arr, format='JPEG', quality=90)
        images_bytes.append(img_byte_arr.getvalue())
        frame_indices.append(frame_idx)
        frame_timestamps_ms.append(timestamp_ms)

    vidcap.release()
    return images_bytes, frame_indices, frame_timestamps_ms, fps, video_duration_ms


def process_single_video(args_tuple):
    """Process one video → dict ready for WebDataset writing."""
    video_path, labels, identity, num_frames, im_res, num_mel_bins, max_audio_length = args_tuple

    video_id = Path(video_path).stem
    result = {'video_id': video_id, 'valid': False, 'identity': identity}

    # Audio
    try:
        fbank, fbank_length = extract_audio_fbank(video_path, num_mel_bins, max_audio_length)
        result['fbank'] = fbank
        result['fbank_length'] = fbank_length
    except Exception as e:
        logging.warning(f"Audio error for {video_id}: {e}")
        return result

    # Video
    try:
        extract_fn = extract_frames_decord if HAS_DECORD else extract_frames_opencv
        images_bytes, frame_indices, frame_timestamps_ms, fps, duration_ms = extract_fn(
            video_path, num_frames, im_res, max_audio_length
        )
        result['images'] = images_bytes
        result['frame_indices'] = frame_indices
        result['frame_timestamps_ms'] = frame_timestamps_ms
        result['video_fps'] = fps
        result['video_duration_ms'] = duration_ms
        result['labels'] = labels
        result['valid'] = True
    except Exception as e:
        logging.warning(f"Video error for {video_id}: {e}")
        return result

    return result


# ---------------------------------------------------------------------------
# WebDataset shard writing
# ---------------------------------------------------------------------------

def write_webdataset_shards(
    samples: list,
    output_dir: str,
    split_name: str,
    max_shard_size: int = 1_000_000_000,  # ~1GB
    shard_prefix: str = "shard",
):
    """Write processed samples to WebDataset .tar shards."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    pattern = os.path.join(split_dir, f"{shard_prefix}-%06d.tar")
    sink = wds.ShardWriter(pattern, maxsize=max_shard_size)

    sample_count = 0
    for sample in samples:
        if not sample['valid']:
            continue

        key = f"{sample_count:08d}"

        # Build the WebDataset record
        record = {"__key__": key}

        # Metadata as JSON
        metadata = {
            'video_id': sample['video_id'],
            'fbank_length': sample['fbank_length'],
            'frame_indices': sample['frame_indices'],
            'frame_timestamps_ms': sample['frame_timestamps_ms'],
            'video_fps': sample['video_fps'],
            'video_duration_ms': sample['video_duration_ms'],
            'identity': sample.get('identity', ''),
        }
        if sample.get('labels') is not None:
            metadata['labels'] = sample['labels']

        record['json'] = json.dumps(metadata).encode('utf-8')

        # Fbank as numpy (preserves float16 dtype)
        record['fbank.npy'] = sample['fbank'].numpy()

        # JPEG frames
        for i, img_bytes in enumerate(sample['images']):
            record[f'{i:03d}.jpg'] = img_bytes

        sink.write(record)
        sample_count += 1

    sink.close()
    n_shards = len([f for f in os.listdir(split_dir) if f.endswith('.tar')])
    logging.info(f"Wrote {sample_count} samples to {split_dir}/ ({n_shards} shards)")
    return sample_count


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_input_file(input_file: str) -> list:
    """Load input CSV or JSON → list of dicts with 'video_path' and optional 'labels'."""
    if input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)['data']
        return [
            {'video_path': item.get('video_path') or item.get('video_id'),
             'labels': item.get('labels')}
            for item in data
        ]
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        path_col = 'video_name' if 'video_name' in df.columns else 'video_path'
        label_col = next((c for c in ['labels', 'target', 'label'] if c in df.columns), None)
        return [
            {'video_path': row[path_col],
             'labels': row[label_col] if label_col else None}
            for _, row in df.iterrows()
        ]
    else:
        raise ValueError(f"Unknown input format: {input_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create WebDataset shards with identity-aware splitting")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV or JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shards")

    # Identity-aware splitting
    parser.add_argument("--split_by_identity", action='store_true',
                        help="Split data by identity into train/val/test")
    parser.add_argument("--split_ratios", type=float, nargs=3, default=[0.9, 0.05, 0.05],
                        help="Train/val/test ratios (default: 0.9 0.05 0.05)")
    parser.add_argument("--exclude_identities_file", type=str, default=None,
                        help="Text file with identity IDs to exclude (one per line)")
    parser.add_argument("--split_name", type=str, default=None,
                        help="If not splitting, use this as the split directory name (e.g., 'test')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")

    # Processing params (match create_sharded_dataset.py defaults)
    parser.add_argument("--total_frame", type=int, default=16)
    parser.add_argument("--im_res", type=int, default=224)
    parser.add_argument("--max_audio_length", type=int, default=1024)
    parser.add_argument("--num_mel_bins", type=int, default=128)
    parser.add_argument("--max_shard_size", type=int, default=1_000_000_000, help="Max shard size in bytes (~1GB)")

    # Parallelism
    try:
        default_workers = len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    parser.add_argument("--num_workers", type=int, default=default_workers)

    args = parser.parse_args()

    # Load input
    file_list = load_input_file(args.input_file)
    logging.info(f"Loaded {len(file_list)} videos from {args.input_file}")

    # Load excluded identities
    exclude_identities = set()
    if args.exclude_identities_file:
        with open(args.exclude_identities_file) as f:
            exclude_identities = {line.strip() for line in f if line.strip()}
        logging.info(f"Loaded {len(exclude_identities)} identities to exclude")

    # Split or use as-is
    if args.split_by_identity:
        splits = create_identity_splits(
            file_list, tuple(args.split_ratios), exclude_identities, args.seed
        )
    else:
        split_name = args.split_name or 'train'
        # Still extract identity for metadata
        for item in file_list:
            item['identity'] = extract_identity(item['video_path'])
        splits = {split_name: file_list}

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split
    total_metadata = {}
    for split_name, split_files in splits.items():
        if not split_files:
            logging.warning(f"Split '{split_name}' is empty, skipping")
            continue

        logging.info(f"Processing split '{split_name}': {len(split_files)} videos")

        # Build task args
        task_args = [
            (item['video_path'], item.get('labels'), item.get('identity', ''),
             args.total_frame, args.im_res, args.num_mel_bins, args.max_audio_length)
            for item in split_files
        ]

        # Process videos in parallel
        valid_samples = []
        with Pool(args.num_workers) as pool:
            for res in tqdm(pool.imap(process_single_video, task_args),
                           total=len(task_args), desc=f"Processing {split_name}",
                           mininterval=10, file=sys.stdout):
                if res['valid']:
                    valid_samples.append(res)

        if not valid_samples:
            logging.warning(f"No valid samples for split '{split_name}'!")
            continue

        # Write WebDataset shards
        n_written = write_webdataset_shards(
            valid_samples, args.output_dir, split_name,
            max_shard_size=args.max_shard_size,
        )

        total_metadata[split_name] = {
            'total_input': len(split_files),
            'total_valid': n_written,
            'failed': len(split_files) - n_written,
        }

    # Save metadata
    metadata = {
        'splits': total_metadata,
        'args': vars(args),
        'decoder': 'decord' if HAS_DECORD else 'opencv',
    }
    meta_path = os.path.join(args.output_dir, 'webdataset_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Metadata saved to {meta_path}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
