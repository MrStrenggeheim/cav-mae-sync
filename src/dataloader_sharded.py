
import os
import torch
import torch.utils.data
from torch.utils.data import IterableDataset
import glob
import random
import logging
import io
from PIL import Image
import torchvision.transforms as T
import math
import numpy as np

class ShardedAudiosetDataset(IterableDataset):
    def __init__(self, shard_dir, audio_conf, shuffle_shards=True, use_mmap=False, dataset_fraction=1.0):
        """
        Args:
            shard_dir (str): Directory containing .pt shards
            audio_conf (dict): Configuration dictionary (matching AudiosetDataset)
            shuffle_shards (bool): Whether to shuffle list of shards
            use_mmap (bool): Use memory-mapped I/O for loading shards. 
                             Default False (safer for network storage like NFS/Lustre).
                             Set True only for local SSD/NVMe storage.
            dataset_fraction (float): Fraction of dataset to use (0.0-1.0). 
                                      Useful for fast testing. Default 1.0 (full dataset).
        """
        self.shard_dir = shard_dir
        self.audio_conf = audio_conf
        self.shuffle_shards = shuffle_shards
        self.use_mmap = use_mmap
        self.dataset_fraction = max(0.0, min(1.0, dataset_fraction))  # Clamp to [0, 1]
        
        all_shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
        if not all_shards:
            raise FileNotFoundError(f"No shards found in {shard_dir}")
        
        # Apply dataset_fraction by selecting a subset of shards
        if self.dataset_fraction < 1.0:
            num_shards_to_use = max(1, int(len(all_shards) * self.dataset_fraction))
            self.shards = all_shards[:num_shards_to_use]
            logging.info(f"Using {self.dataset_fraction*100:.1f}% of dataset: {len(self.shards)}/{len(all_shards)} shards")
        else:
            self.shards = all_shards
            
        logging.info(f"Found {len(self.shards)} shards.")
        
        self.im_res = audio_conf.get('im_res', 224)
        self.norm_mean = audio_conf.get('mean')
        self.norm_std = audio_conf.get('std')
        self.skip_norm = audio_conf.get('skip_norm', False)
        self.target_length = audio_conf.get('target_length', 48)
        self.num_mel_bins = audio_conf.get('num_mel_bins', 128)
        
        # Standard ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.4850, 0.4560, 0.4060],
            std=[0.2290, 0.2240, 0.2250]
        )
        
        # Augmentation (if enabled)
        self.augmentation = audio_conf.get('augmentation', False)
        if self.augmentation:
            self.transform = T.Compose([
                T.RandomResizedCrop(self.im_res, scale=(0.08, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                self.normalize
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                self.normalize
            ])
        
        # Compute approximate length by reading first shard and multiplying
        # (assumes all shards have roughly equal size)
        self._total_samples = 0
        if self.shards:
            try:
                first_shard = torch.load(self.shards[0], weights_only=False)
                samples_per_shard = len(first_shard)
                self._total_samples = samples_per_shard * len(self.shards)
            except Exception:
                pass
        logging.info(f"Estimated total samples in dataset: {self._total_samples}")
    
    def __len__(self):
        # In DDP, each rank only sees a subset of shards
        if torch.distributed.is_initialized():
            return self._total_samples // torch.distributed.get_world_size()
        return self._total_samples

    def slice_fbank_at_timestamp(self, full_fbank, fbank_length, timestamp_ms, target_length):
        """
        Slice fbank centered at timestamp with edge padding.
        fbank frame_shift = 10ms, so timestamp_ms / 10 = center frame index.
        Always returns tensor of shape [target_length, 128].
        """
        FRAME_SHIFT_MS = 10
        center_frame = int(timestamp_ms / FRAME_SHIFT_MS)
        half_len = target_length // 2
        
        start = center_frame - half_len
        end = start + target_length
        
        # Clamp to actual fbank length (not padded storage length)
        actual_length = fbank_length
        
        # Compute padding needed at each edge
        pad_left = 0
        pad_right = 0
        
        if start < 0:
            pad_left = -start
            start = 0
        if end > actual_length:
            pad_right = end - actual_length
            end = actual_length
        
        # Extract segment
        segment = full_fbank[start:end, :]
        
        # Pad if needed (at edges of audio)
        if pad_left > 0 or pad_right > 0:
            segment = torch.nn.functional.pad(segment, (0, 0, pad_left, pad_right))
        
        if segment.shape[0] != target_length:
            original_shape = segment.shape[0]
            if segment.shape[0] < target_length:
                pad_needed = target_length - segment.shape[0]
                segment = torch.nn.functional.pad(segment, (0, 0, 0, pad_needed))
            else:
                segment = segment[:target_length, :]
            logging.warning(f"Segment shape mismatch: got {original_shape}, expected {target_length}, corrected")
        return segment

    def flatten_dataset(self, data):
        # Data is dict: {'video_id', 'fbank', 'fbank_length', 'images', 'frame_indices', 'frame_timestamps_ms', ...}
        # fbank: Tensor [max_audio_length, 128] float16 (padded)
        # fbank_length: int (actual length before padding)
        # images: List[bytes] (JPEG)
        
        full_fbank = data['fbank'].float()
        fbank_length = data.get('fbank_length', full_fbank.shape[0])  # Fallback for old shards
        images_list = data['images']
        frame_timestamps_ms = data.get('frame_timestamps_ms', None)
        frame_indices = data['frame_indices']
        
        fbanks = []
        images = []
        
        target_length = self.target_length
        
        for i, frame_idx in enumerate(frame_indices):
            # Use timestamp-based slicing if available, else fallback to old method
            if frame_timestamps_ms is not None:
                timestamp_ms = frame_timestamps_ms[i]
                fbank = self.slice_fbank_at_timestamp(
                    full_fbank, fbank_length, timestamp_ms, target_length
                )
            else:
                # Legacy fallback for old shards
                logging.warning("Using legacy fallback for old shards")
                num_frames_conf = self.audio_conf.get('total_frame', 16)
                spectrogram_length = fbank_length
                frame_position = int(round(i * spectrogram_length / num_frames_conf))
                start = max(0, frame_position - target_length // 2)
                end = start + target_length
                if end > spectrogram_length:
                    end = spectrogram_length
                    start = max(0, end - target_length)
                fbank = full_fbank[start:end, :]
                # Ensure exact target_length (pad or trim)
                if fbank.shape[0] < target_length:
                    pad_len = target_length - fbank.shape[0]
                    fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad_len))
                elif fbank.shape[0] > target_length:
                    fbank = fbank[:target_length, :]
            
            # Normalize Audio
            if not self.skip_norm:
                 fbank = (fbank - self.norm_mean) / self.norm_std

            # Defensive shape validation to prevent torch.stack failures
            expected_shape = (target_length, self.num_mel_bins)
            if fbank.shape != expected_shape:
                # Fix shape by padding/truncating
                t, f = fbank.shape
                if f != self.num_mel_bins:
                    if f < self.num_mel_bins:
                        fbank = torch.nn.functional.pad(fbank, (0, self.num_mel_bins - f))
                    else:
                        fbank = fbank[:, :self.num_mel_bins]
                if t != target_length:
                    if t < target_length:
                        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - t))
                    else:
                        fbank = fbank[:target_length, :]

            fbanks.append(fbank)
            
            try:
                img_bytes = images_list[i]
                img = Image.open(io.BytesIO(img_bytes))
                images.append(self.transform(img))
            except Exception as e:
                logging.warning(f"Error decoding image: {e}")
                images.append(torch.zeros(3, self.im_res, self.im_res))

        if fbanks:
            fbanks = torch.stack(fbanks)
            images = torch.stack(images)
        else:
             fbanks = torch.zeros(len(frame_indices), self.target_length, self.num_mel_bins)
             images = torch.zeros(len(frame_indices), 3, self.im_res, self.im_res)

        return fbanks, images, data['video_id'], torch.tensor(frame_indices)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            per_rank = int(math.ceil(len(self.shards) / float(world_size)))
            rank_start = rank * per_rank
            rank_end = min(rank_start + per_rank, len(self.shards))
            shards_for_rank = self.shards[rank_start:rank_end]
            logging.info(f"Rank {rank}/{world_size}: assigned shards [{rank_start}:{rank_end}] ({len(shards_for_rank)} shards)")
        else:
            shards_for_rank = self.shards
        
        if worker_info is None:
            shards_to_read = shards_for_rank
        else:
            per_worker = int(math.ceil(len(shards_for_rank) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(shards_for_rank))
            shards_to_read = shards_for_rank[iter_start:iter_end]
            logging.info(f"Worker {worker_info.id}/{worker_info.num_workers}: assigned {len(shards_to_read)} shards")
            
        if self.shuffle_shards:
            random.shuffle(shards_to_read)
            
        for shard_path in shards_to_read:
            try:
                data_list = torch.load(shard_path, weights_only=False, mmap=self.use_mmap)
            except Exception as e:
                logging.warning(f"Error loading shard {shard_path}: {e}")
                continue
                
            if self.shuffle_shards:  # Shuffle samples within shard
                random.shuffle(data_list)
                
            for item in data_list:
                try:
                    yield self.flatten_dataset(item)
                except Exception as e:
                    video_id = item.get('video_id', 'unknown')
                    logging.warning(f"Error processing sample {video_id} in {shard_path}: {e}")
                    # Skip this sample, continue with next
                    continue
