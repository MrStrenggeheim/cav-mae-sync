
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchvision.transforms as T
import cv2
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import json
import logging
import io
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_video_id(input_f):
    """Extracts video ID from filename."""
    try:
        parts = input_f.split('/')
        filename = parts[-1]
        ext_len = len(filename.split('.')[-1])
        return "-".join(parts[-5:])[:-ext_len-1]
    except Exception:
        return os.path.splitext(os.path.basename(input_f))[0]

def map_frame_to_spectrogram(frame_index, num_frames, spectrogram_length, target_length):
    """
    Maps a frame index to a corresponding segment in the spectrogram.
    """
    frame_position = int(round(frame_index * spectrogram_length / num_frames))
    
    start = max(0, frame_position - target_length // 2)
    end = start + target_length
    
    if end > spectrogram_length:
        end = spectrogram_length
        start = max(0, end - target_length)
    
    return (start, end)

def _wav2fbank(waveform, sr, target_length=1024, num_mel_bins=128):
    """
    Computes fbank features from waveform.
    """
    waveform = waveform - waveform.mean()
    
    # fbank computation
    try:
        fbank = kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, 
                           use_energy=False, window_type='hanning', 
                           num_mel_bins=num_mel_bins, dither=0.0, frame_shift=10)
    except Exception as e:
        logging.warning(f"Fbank computation failed: {e}")
        return torch.zeros([target_length, num_mel_bins])

    n_frames = fbank.shape[0]
    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    return fbank

def process_single_video(args):
    """
    Worker function to process a single video.
    Extracts 16 specific frames (JPEG bytes) and the FULL audio spectrogram (Float16).
    Returns:
      images: List[bytes] (JPEG)
      fbank: Tensor [Time, 128] (float16)
    """
    video_path, labels, target_length, num_frames, im_res, num_mel_bins, max_audio_length = args
    
    video_id = get_video_id(video_path)
    result = {
        'video_id': video_id,
        'video_path': video_path, 
        'labels': labels,
        'valid': False
    }

    # Audio Processing
    try:
        try:
            waveform, sr = torchaudio.load(video_path)
            
            # Mix to mono
            if waveform.shape[0] > 1:
                # select ch0 (TODO: Do we really only want to use ch0?)
                waveform = waveform[0:1, :]
                
            if sr != 16000:
                 resampler = torchaudio.transforms.Resample(sr, 16000)
                 waveform = resampler(waveform)
                 sr = 16000
                 
        except Exception as e:
            logging.warning(f"torchaudio.load failed for {video_path}, falling back to ffmpeg CLI: {e}")
            command = [
                'ffmpeg', '-v', 'quiet', '-i', video_path, 
                '-f', 's16le', '-ar', '16000', '-ac', '1', 
                '-af', 'pan=mono|c0=c0', '-'
            ]
            pipe = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            raw_audio = np.frombuffer(pipe.stdout, dtype=np.int16)
            waveform = torch.from_numpy(raw_audio).float().unsqueeze(0) / 32768.0
            sr = 16000
            
        # Compute Full Fbank for the video
        waveform = waveform - waveform.mean()
        full_fbank = kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, 
                           use_energy=False, window_type='hanning', 
                           num_mel_bins=num_mel_bins, dither=0.0, frame_shift=10)
        
        # Enforce strict frame length (1024)
        n_frames = full_fbank.shape[0]
        p = max_audio_length - n_frames

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            full_fbank = m(full_fbank)
        elif p < 0:
            full_fbank = full_fbank[0:max_audio_length, :]
        
        # Store processed fbank as FLOAT16
        result['fbank'] = full_fbank.half()
                           
    except Exception as e:
         logging.warning(f"Audio processing error for {video_id}: {e}")
         return result

    # Video Processing
    try:
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            logging.warning(f"Could not open video {video_path}")
            return result
            
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))
        
        images_bytes = []
        frame_indices = []
        
        pil_transform = T.Compose([
             T.Resize(im_res),
             T.CenterCrop(im_res)
        ])
        
        for i in range(num_frames):
            frame_idx = int(i * (total_frame_num/num_frames))
            
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # We initially had frame_idx - 1 here, but this is not correct
            ret, frame = vidcap.read()
            
            if ret:
                cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                
                # Resize and Crop
                pil_im = pil_transform(pil_im)
                
                # Encode to JPEG Bytes
                img_byte_arr = io.BytesIO()
                pil_im.save(img_byte_arr, format='JPEG', quality=90)
                images_bytes.append(img_byte_arr.getvalue())
                
                frame_indices.append(frame_idx)
            else:
                 logging.warning(f"Could not read frame {frame_idx} for {video_id}")
                 # Backup: zero image
                 black_im = Image.new('RGB', (im_res, im_res))
                 img_byte_arr = io.BytesIO()
                 black_im.save(img_byte_arr, format='JPEG')
                 images_bytes.append(img_byte_arr.getvalue())
                 
                 frame_indices.append(frame_idx) 
                 vidcap.release()
                 return result

        vidcap.release()
        
        # Store LIST of bytes
        result['images'] = images_bytes
        result['frame_indices'] = frame_indices
        result['valid'] = True
        
    except Exception as e:
        logging.warning(f"Video processing error for {video_id}: {e}")
        return result

    return result

def main():
    parser = argparse.ArgumentParser(description="Create Sharded Dataset (Optimized: Float16 Fbank + JPEG)")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV/JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shards")
    parser.add_argument("--num_shards", type=int, default=100, help="Target number of shards")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--total_frame", type=int, default=16, help="Frames to extract")
    parser.add_argument("--im_res", type=int, default=224, help="Image resolution")
    parser.add_argument("--target_length", type=int, default=416, help="Target audio length (for reference/slicing in dataloader)")
    parser.add_argument("--max_audio_length", type=int, default=1024, help="Max audio length to store (legacy truncation/pad limit)")
    parser.add_argument("--num_mel_bins", type=int, default=128, help="Mel bins")
    
    args = parser.parse_args()

    # Load Data
    if args.input_file.endswith('.json'):
        with open(args.input_file, 'r') as f:
            data = json.load(f)['data']
            file_list = []
            for item in data:
               file_list.append({
                   'video_path': item.get('video_path') or item.get('video_id'),
                   'labels': item.get('labels'),
                   'wav': item.get('wav')
               })
    elif args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
        col_name = 'video_name' if 'video_name' in df.columns else 'video_path'
        file_list = [{'video_path': row[col_name], 'labels': row.get('labels', None)} for _, row in df.iterrows()]
    else:
        raise ValueError("Unknown input file format")
    
    total_files = len(file_list)
    logging.info(f"Found {total_files} files to process.")
    
    shard_size = int(np.ceil(total_files / args.num_shards))
    logging.info(f"Shard size: ~{shard_size} videos/shard")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for shard_id in range(args.num_shards):
        start_idx = shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_files)
        
        if start_idx >= total_files:
            break
            
        shard_data = file_list[start_idx:end_idx]
        shard_output_path = os.path.join(args.output_dir, f"shard_{shard_id:04d}.pt")
        
        if os.path.exists(shard_output_path):
             logging.info(f"Shard {shard_id} exists. Skipping.")
             continue
        
        logging.info(f"Processing Shard {shard_id} ({len(shard_data)} files)...")
        
        task_args = []
        for item in shard_data:
            path = item.get('video_path')
            labels = item.get('labels')
            task_args.append((path, labels, args.target_length, args.total_frame, args.im_res, args.num_mel_bins, args.max_audio_length))
            
        valid_samples = []
        with Pool(args.num_workers) as pool:
            for res in tqdm(pool.imap(process_single_video, task_args), total=len(shard_data)):
                if res['valid']:
                    valid_samples.append(res)
        
        if len(valid_samples) > 0:
            logging.info(f"Saving {len(valid_samples)} samples to {shard_output_path}")
            torch.save(valid_samples, shard_output_path)
        else:
            logging.warning(f"No valid samples for shard {shard_id}!")

    # Save metadata
    metadata = {
        'num_shards': args.num_shards,
        'total_files': total_files,
        'args': vars(args)
    }
    with open(os.path.join(args.output_dir, 'sharded_dataset.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    logging.info("Done.")

if __name__ == "__main__":
    main()
