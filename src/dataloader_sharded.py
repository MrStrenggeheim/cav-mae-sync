
import os
import json
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
    def __init__(self, shard_dir, audio_conf, shuffle_shards=True, use_mmap=False, dataset_fraction=1.0,
                 rank=None, world_size=None, max_samples=None):
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
            rank (int, optional): DDP rank. If None, auto-detected from env vars or torch.distributed.
            world_size (int, optional): DDP world size. If None, auto-detected.
            max_samples (int, optional): Maximum samples to yield per epoch. Used to ensure
                                         equal iteration length across DDP ranks. If None, no limit.
        """
        self.max_samples = max_samples
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
        
        # Determine rank and world_size for DDP
        # Priority: explicit args > torch.distributed > env vars > defaults
        if rank is not None and world_size is not None:
            self._rank = rank
            self._world_size = world_size
        elif torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            # Fallback to environment variables (set by torchrun/DDP launcher)
            # This is the standard pattern for IterableDataset + DDP
            self._rank = int(os.environ.get('RANK', 0))
            self._world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Load shard metadata if available (for efficient sample counting)
        self._shard_metadata = self._load_shard_metadata()
        logging.info(f"Dataset initialized for rank {self._rank}/{self._world_size} (max_samples={self.max_samples}, has_metadata={self._shard_metadata is not None})")
    
    def _load_shard_metadata(self):
        """Load per-shard sample counts from metadata file if available."""
        metadata_path = os.path.join(self.shard_dir, 'sharded_dataset.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if 'shard_sample_counts' in metadata:
                    logging.info(f"Loaded shard metadata with {len(metadata['shard_sample_counts'])} shard counts")
                    return metadata['shard_sample_counts']
            except Exception as e:
                logging.warning(f"Failed to load shard metadata: {e}")
        return None
    
    def __len__(self):
        # In DDP, each rank only sees a subset of shards
        if self.max_samples is not None:
            return self.max_samples
        return self._total_samples // self._world_size

    def count_samples_for_rank(self):
        """
        Count actual number of samples in shards assigned to this rank.
        Uses metadata if available, otherwise falls back to loading shards (slow).

        Returns:
            int: Total samples in this rank's shards
        """
        # Compute which shards this rank owns
        if self._world_size > 1:
            per_rank = int(math.ceil(len(self.shards) / float(self._world_size)))
            rank_start = self._rank * per_rank
            rank_end = min(rank_start + per_rank, len(self.shards))
            shards_for_rank = self.shards[rank_start:rank_end]
        else:
            shards_for_rank = self.shards

        # Use metadata if available (O(1) per shard, no memory load)
        if self._shard_metadata is not None:
            total = 0
            missing_metadata = []
            for shard_path in shards_for_rank:
                shard_name = os.path.basename(shard_path)
                if shard_name in self._shard_metadata:
                    total += self._shard_metadata[shard_name]
                else:
                    missing_metadata.append(shard_name)
            
            if missing_metadata:
                logging.warning(f"Missing metadata for {len(missing_metadata)} shards, loading them to count (slow)")
                for shard_path in shards_for_rank:
                    if os.path.basename(shard_path) in missing_metadata:
                        try:
                            data_list = torch.load(shard_path, weights_only=False)
                            total += len(data_list)
                        except Exception as e:
                            logging.warning(f"Error counting samples in {shard_path}: {e}")
            return total

        # Fallback: no metadata, load all shards (SLOW - logs warning)
        logging.warning(
            "No shard metadata found! Loading ALL shards to count samples. "
            "This is SLOW and memory-intensive. Run: python scripts/generate_shard_metadata.py --shard_dir <path>"
        )
        total = 0
        for shard_path in shards_for_rank:
            try:
                data_list = torch.load(shard_path, weights_only=False, mmap=self.use_mmap)
                total += len(data_list)
            except Exception as e:
                logging.warning(f"Error counting samples in {shard_path}: {e}")
        return total

    def set_max_samples(self, max_samples):
        """Set the maximum samples per epoch (used for DDP sync)."""
        self.max_samples = max_samples
        logging.info(f"Rank {self._rank}: set max_samples={max_samples}")

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
                logging.warning(f"Shape mismatch in {data['video_id']}: got {fbank.shape}, expected {expected_shape}. Correcting.")
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


    def _create_dummy_sample(self, video_id="dummy_error"):
        """Create a zero-filled sample to replace corrupted data (maintains DDP sync)."""
        num_frames = self.audio_conf.get('total_frame', 16)
        target_length = self.target_length
        im_res = self.im_res
        
        fbanks = torch.zeros(num_frames, target_length, self.num_mel_bins)
        images = torch.zeros(num_frames, 3, im_res, im_res)
        frame_indices = torch.arange(num_frames)
        
        return fbanks, images, video_id, frame_indices

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        rank = self._rank
        world_size = self._world_size

        # Partition shards by rank (DDP)
        if world_size > 1:
            per_rank = int(math.ceil(len(self.shards) / float(world_size)))
            rank_start = rank * per_rank
            rank_end = min(rank_start + per_rank, len(self.shards))
            shards_for_rank = self.shards[rank_start:rank_end]
            logging.info(f"Rank {rank}/{world_size}: assigned shards [{rank_start}:{rank_end}] ({len(shards_for_rank)} shards)")
        else:
            shards_for_rank = self.shards

        # Worker info
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # PARTITIONED LOADING: Each worker gets a subset of this rank's shards
        # This prevents I/O contention where all workers fight for the same files
        per_worker = len(shards_for_rank) // num_workers
        worker_start = worker_id * per_worker
        if worker_id == num_workers - 1:
            # Last worker gets remaining shards
            worker_end = len(shards_for_rank)
        else:
            worker_end = worker_start + per_worker
        
        shards_for_worker = shards_for_rank[worker_start:worker_end]
        logging.info(f"[Worker {worker_id}/{num_workers}] Assigned shards [{worker_start}:{worker_end}] ({len(shards_for_worker)} of {len(shards_for_rank)} rank shards)")

        if self.shuffle_shards:
            random.shuffle(shards_for_worker)

        # Each worker yields ALL samples from its assigned shards - no interleaving
        sample_idx = 0
        for shard_idx, shard_path in enumerate(shards_for_worker):
            load_start = time.time()
            print(f"[Worker {worker_id}] Loading shard {shard_idx+1}/{len(shards_for_worker)}: {os.path.basename(shard_path)}", flush=True)
            
            try:
                data_list = torch.load(shard_path, weights_only=False, mmap=self.use_mmap)
                load_duration = time.time() - load_start
                print(f"[Worker {worker_id}] Loaded {os.path.basename(shard_path)} in {load_duration:.2f}s ({len(data_list)} items)", flush=True)
                
                # Warn if load took too long (potential NFS issue)
                if load_duration > 30:
                    logging.warning(f"[Worker {worker_id}] SLOW SHARD LOAD: {shard_path} took {load_duration:.1f}s (>30s)")
                    
            except Exception as e:
                logging.warning(f"Error loading shard {shard_path}: {e}")
                continue

            if self.shuffle_shards:
                random.shuffle(data_list)

            for item in data_list:
                # Check max_samples limit (for DDP sync)
                if self.max_samples is not None and sample_idx >= self.max_samples:
                    return

                try:
                    yield self.flatten_dataset(item)
                except Exception as e:
                    video_id = item.get('video_id', 'unknown')
                    error_msg = f"Error processing sample {video_id} in {shard_path}: {e}"
                    logging.warning(f" [WARNING: DUMMY SAMPLE] {error_msg}. Yielding ZERO-FILLED dummy sample to maintain DDP synchronization!")
                    yield self._create_dummy_sample(f"corrupt_{video_id}")

                sample_idx += 1

