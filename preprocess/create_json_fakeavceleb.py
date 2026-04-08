#!/usr/bin/env python3
"""
Create JSON manifest for FakeAVCeleb dataset evaluation.

This script generates the JSON format required by AudiosetDataset from FakeAVCeleb's
preprocessed data (extracted frames + audio).

Expected input directory structure:
    DATA_PATH/
        audio/          # WAV files named by video_id (e.g., id00001-video001.wav)
        frames/         # Subdirectories frame_0/, frame_1/, ... with JPG files
        labels.csv      # Optional: video_id,label (0=real, 1=fake)

Output JSON format:
    {"data": [
        {"video_id": "...", "wav": "path/to/audio.wav", "video_path": "path/to/frames", "labels": "0"},
        ...
    ]}

Usage:
    python create_json_fakeavceleb.py --data_path /path/to/fakeavceleb/preprocessed --output_json fakeavceleb.json
"""

import argparse
import glob
import json
import logging
import os
import pandas as pd
from pathlib import Path


def find_video_ids_from_audio(audio_dir: str) -> list:
    """Find all video IDs from audio WAV files."""
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    video_ids = []
    for wav_path in wav_files:
        # Extract video_id from filename (remove .wav extension)
        video_id = Path(wav_path).stem
        video_ids.append(video_id)
    return sorted(video_ids)


def find_video_ids_from_frames(frames_dir: str) -> list:
    """Find all video IDs from frame directories."""
    # Look for JPG files in frame_0/ subdirectory
    frame_0_dir = os.path.join(frames_dir, "frame_0")
    if not os.path.exists(frame_0_dir):
        logging.warning(f"frame_0 directory not found at {frame_0_dir}")
        return []
    
    jpg_files = glob.glob(os.path.join(frame_0_dir, "*.jpg"))
    video_ids = []
    for jpg_path in jpg_files:
        video_id = Path(jpg_path).stem
        video_ids.append(video_id)
    return sorted(video_ids)


def create_json_manifest(
    data_path: str,
    output_json: str,
    labels_csv: str = None,
    label_col: str = "label"
) -> dict:
    """
    Create JSON manifest for FakeAVCeleb dataset.
    
    Args:
        data_path: Root directory with audio/ and frames/ subdirectories
        output_json: Path to output JSON file
        labels_csv: Optional CSV with video_id,label columns
        label_col: Name of label column in CSV (default: "label")
        
    Returns:
        Statistics dict with counts
    """
    audio_dir = os.path.join(data_path, "audio")
    frames_dir = os.path.join(data_path, "frames")
    
    # Validate directories exist
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Find video IDs
    audio_ids = set(find_video_ids_from_audio(audio_dir))
    frame_ids = set(find_video_ids_from_frames(frames_dir))
    
    # Use intersection (videos with both audio and frames)
    common_ids = audio_ids & frame_ids
    audio_only = audio_ids - frame_ids
    frames_only = frame_ids - audio_ids
    
    if audio_only:
        logging.warning(f"Found {len(audio_only)} videos with audio but no frames")
    if frames_only:
        logging.warning(f"Found {len(frames_only)} videos with frames but no audio")
    
    logging.info(f"Found {len(common_ids)} valid videos (with both audio and frames)")
    
    # Load labels if provided
    labels_dict = {}
    if labels_csv and os.path.exists(labels_csv):
        df = pd.read_csv(labels_csv)
        if 'video_id' in df.columns and label_col in df.columns:
            labels_dict = dict(zip(df['video_id'], df[label_col]))
            logging.info(f"Loaded {len(labels_dict)} labels from {labels_csv}")
        else:
            logging.warning(f"Labels CSV missing required columns. Found: {list(df.columns)}")
    
    # Build data list
    data = []
    for video_id in sorted(common_ids):
        item = {
            "video_id": video_id,
            "wav": os.path.join(audio_dir, f"{video_id}.wav"),
            "video_path": frames_dir,
            "labels": str(labels_dict[video_id]),  # Require explicit label — KeyError if missing
        }
        data.append(item)
    
    # Write JSON
    output = {"data": data}
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    logging.info(f"Created JSON manifest with {len(data)} videos: {output_json}")
    
    # Count labels
    if labels_dict:
        real_count = sum(1 for v in data if v["labels"] == "0")
        fake_count = sum(1 for v in data if v["labels"] == "1")
        logging.info(f"Label distribution: {real_count} real, {fake_count} fake")
    
    return {
        "total_videos": len(data),
        "audio_only": len(audio_only),
        "frames_only": len(frames_only),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Create JSON manifest for FakeAVCeleb dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to preprocessed FakeAVCeleb data (contains audio/ and frames/ dirs)"
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--labels_csv", type=str, default=None,
        help="Optional CSV with video_id,label columns"
    )
    parser.add_argument(
        "--label_col", type=str, default="label",
        help="Name of the label column in the CSV (default: label)"
    )
    
    args = parser.parse_args()
    
    stats = create_json_manifest(
        data_path=args.data_path,
        output_json=args.output_json,
        labels_csv=args.labels_csv,
        label_col=args.label_col
    )
    
    logging.info(f"Done! Stats: {stats}")


if __name__ == "__main__":
    main()
