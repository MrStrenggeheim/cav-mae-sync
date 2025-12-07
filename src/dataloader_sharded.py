
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
    def __init__(self, shard_dir, audio_conf, shuffle_shards=True):
        """
        Args:
            shard_dir (str): Directory containing .pt shards
            audio_conf (dict): Configuration dictionary (matching AudiosetDataset)
            shuffle_shards (bool): Whether to shuffle list of shards
        """
        self.shard_dir = shard_dir
        self.audio_conf = audio_conf
        self.shuffle_shards = shuffle_shards
        
        self.shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
        if not self.shards:
            raise FileNotFoundError(f"No shards found in {shard_dir}")
            
        logging.info(f"Found {len(self.shards)} shards.")
        
        self.im_res = audio_conf.get('im_res', 224)
        self.norm_mean = audio_conf.get('mean')
        self.norm_std = audio_conf.get('std')
        self.skip_norm = audio_conf.get('skip_norm', False)
        self.target_length = audio_conf.get('target_length', 416)
        
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

    def map_frame_to_spectrogram(self, frame_index, num_frames, spectrogram_length, target_length):
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

    def flatten_dataset(self, data):
        # Data is dict: {'video_id', 'fbank', 'images', 'frame_indices', ...}
        # fbank: Tensor [Time, 128] float16 (Full Spectrogram)
        # images: List[bytes] (JPEG)
        
        full_fbank = data['fbank'].float()
        images_list = data['images']
        frame_indices = data['frame_indices']
        
        fbanks = []
        images = []
        
        num_frames_conf = self.audio_conf.get('total_frame', 16)
        target_length = self.target_length
        spectrogram_length = full_fbank.shape[0]
        
        for i, frame_idx in enumerate(frame_indices):
            start, end = self.map_frame_to_spectrogram(
                frame_index=i, 
                num_frames=num_frames_conf,
                spectrogram_length=spectrogram_length,
                target_length=target_length
            )
            
            fbank = full_fbank[start:end, :]
            
            # Pad if needed
            n_frames = fbank.shape[0]
            p = target_length - n_frames
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            
            # Normalize Audio
            if not self.skip_norm:
                 fbank = (fbank - self.norm_mean) / self.norm_std

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
             fbanks = torch.zeros(len(frame_indices), self.target_length, 128)
             images = torch.zeros(len(frame_indices), 3, self.im_res, self.im_res)

        return fbanks, images, data['video_id'], torch.tensor(frame_indices)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            logging.info("Using a single process for dataloading.")
            shards_to_read = self.shards
        else:
            logging.info(f"Using {worker_info.num_workers} processes for dataloading.")
            per_worker = int(math.ceil(len(self.shards) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.shards))
            shards_to_read = self.shards[iter_start:iter_end]
            
        if self.shuffle_shards:
            random.shuffle(shards_to_read)
            
        for shard_path in shards_to_read:
            try:
                data_list = torch.load(shard_path)
                if self.shuffle_shards: # Shuffle samples within shard
                    random.shuffle(data_list)
                    
                for item in data_list:
                    yield self.flatten_dataset(item)
                    
            except Exception as e:
                logging.warning(f"Error reading shard {shard_path}: {e}")
                continue
