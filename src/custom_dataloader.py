import os, csv, torch, torchaudio, torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class VideoDataset(Dataset):
    """
    Dataset for multimodal (audio+video) samples stored as MP4 files.
    Each sample yields 16 frames and 16 aligned fbank segments.
    Returns:
        fbanks: [16, 1024, 128]
        frames: [16, 3, 224, 224]
        labels: one-hot [num_classes]
        video_id: str
    """
    def __init__(self, csv_path, audio_conf, num_classes=2):
        self.entries = self._read_csv(csv_path)
        self.audio_conf = audio_conf
        self.num_samples = len(self.entries)
        self.num_classes = num_classes

        # audio/video config
        self.total_frame   = audio_conf.get("total_frame", 16)
        self.target_length = audio_conf.get("target_length", 1024)
        self.melbins       = audio_conf.get("num_mel_bins", 128)
        self.im_res        = audio_conf.get("im_res", 224)
        self.norm_mean     = audio_conf.get("mean", 0)
        self.norm_std      = audio_conf.get("std", 1)
        self.skip_norm     = audio_conf.get("skip_norm", False)

        # transforms
        if audio_conf.get("augmentation", False):
            self.preprocess = T.Compose([
                T.RandomResizedCrop(self.im_res, scale=(0.08, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.preprocess = T.Compose([
                T.Resize(self.im_res),
                T.CenterCrop(self.im_res),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def _read_csv(self, csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            return [(row["video_name"], int(row["target"])) for row in reader]

    def _extract_audio_from_mp4(self, video_path):
        """Extract audio fbank, split into [total_frame, target_length, melbins]."""
        waveform, sr = torchaudio.load(video_path)
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr,
            use_energy=False, window_type="hanning",
            num_mel_bins=self.melbins, dither=0.0, frame_shift=10
        )

        # pad or cut to exactly total_frame * target_length
        n_frames = fbank.shape[0]
        needed = self.target_length * self.total_frame
        if n_frames < needed:
            pad = torch.zeros((needed - n_frames, self.melbins))
            fbank = torch.cat([fbank, pad], dim=0)
        elif n_frames > needed:
            fbank = fbank[:needed, :]

        fbanks = fbank.view(self.total_frame, self.target_length, self.melbins)
        if not self.skip_norm:
            fbanks = (fbanks - self.norm_mean) / self.norm_std
        return fbanks

    def _extract_frames_from_mp4(self, video_path):
        """Extract evenly spaced frames."""
        video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
        total_avail = video.shape[0]
        idxs = np.linspace(0, total_avail - 1, self.total_frame, dtype=int)
        frames = [self.preprocess(Image.fromarray(video[i].numpy())) for i in idxs]
        return torch.stack(frames)

    def __getitem__(self, idx):
        video_path, target = self.entries[idx]
        video_id = video_path

        try:
            fbanks = self._extract_audio_from_mp4(video_path)
        except Exception as e:
            print(f"[WARN] Audio extraction failed for {video_id}: {e}")
            fbanks = torch.zeros((self.total_frame, self.target_length, self.melbins))

        try:
            frames = self._extract_frames_from_mp4(video_path)
        except Exception as e:
            print(f"[WARN] Frame extraction failed for {video_id}: {e}")
            frames = torch.zeros((self.total_frame, 3, self.im_res, self.im_res))

        label = torch.zeros(self.num_classes)
        label[target] = 1.0
        return fbanks, frames, label, video_id

    def __len__(self):
        return self.num_samples
