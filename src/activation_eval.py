import os
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Subset
import random
import pickle as pkl

# repo imports
import models
import custom_dataloader 


class ActivationEvaluator:
    def __init__(self, model, loader, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
        self.model = model.to(self.device).eval()

        self.loader = loader

        # hooks + accumulators
        self.hooks = []
        self.batch_acts = {}

        target = self.model.module
        for name, module in target.named_modules():
            if len(list(module.children())) == 0:
                self.hooks.append(module.register_forward_hook(self._save_activation(name)))

        self.sum_real = defaultdict(float)
        self.sum_fake = defaultdict(float)
        self.cnt_real = defaultdict(int)
        self.cnt_fake = defaultdict(int)

        self.forward_logs = []

    def _save_activation(self, name):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out):
                return
            if out.dim() > 1:
                vals = out.detach().float().view(out.size(0), -1).mean(dim=1)
            else:
                vals = out.detach().float()
            self.batch_acts[name] = vals.cpu()
        return hook

    @staticmethod
    def _labels_to_mask(labels, T=None):
        if labels.ndim == 2:
            is_fake = labels[:, 0] >= 0.5
            is_real = labels[:, 1] >= 0.5
        else:
            is_fake = labels >= 0.5
            is_real = ~is_fake
        if T is not None and T > 1:
            is_fake = is_fake.repeat_interleave(T)
            is_real = is_real.repeat_interleave(T)
        return is_real, is_fake

    def evaluate(self, output_path="."):
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.loader), total=len(self.loader), desc="Evaluating activations"):
                audio, video, labels, video_names = batch
                audio = audio.to(self.device)
                video = video.to(self.device)
                labels = labels.to(self.device)
                
                print(f"Audio shape: {audio.shape}, Video shape: {video.shape}")

                if video.ndim != 5 or audio.ndim != 4:
                    raise RuntimeError(f"Unexpected shapes: audio {tuple(audio.shape)}, video {tuple(video.shape)}")

                # flatten batch so we process all frames at the same time
                a_input = audio.reshape(audio.shape[0] * audio.shape[1], audio.shape[2], audio.shape[3])
                v_input = video.reshape(video.shape[0] * video.shape[1], video.shape[2], video.shape[3], video.shape[4])

                audio_input, video_input = a_input.to(self.device), v_input.to(self.device)

                # Forward
                audio_out, video_out, cls_a, cls_v = self.model.module.forward_feat(audio_input, video_input)
                
                print(f"Audio out shape: {audio_out.shape}, Video out shape: {video_out.shape}")
                print(f"Cls_a shape: {cls_a.shape}, Cls_v shape: {cls_v.shape}")

                B, T, C, H, W = video.shape
                # Log embeddings
                tuples = {
                    "video_name": [],
                    "label": [],
                    "audio_out": [],
                    "video_out": [],
                    "cls_a": [],
                    "cls_v": [],
                }
                for idx in range(B):
                    tuples["video_name"].append(video_names[idx])
                    tuples["label"].append(labels[idx].cpu())
                    tuples["audio_out"].append(audio_out[idx*T:(idx+1)*T].cpu().numpy())
                    tuples["video_out"].append(video_out[idx*T:(idx+1)*T].cpu().numpy())
                    tuples["cls_a"].append(cls_a[idx*T:(idx+1)*T].cpu().numpy())
                    tuples["cls_v"].append(cls_v[idx*T:(idx+1)*T].cpu().numpy())
                pkl.dump(tuples, open(os.path.join(output_path, f"forward_embeddings_{i}.pkl"), "wb"))

                # Masks (CPU)
                is_real, is_fake = self._labels_to_mask(labels, T=T)
                is_real, is_fake = is_real.cpu(), is_fake.cpu()

                # Accumulate activations
                for name, vals in self.batch_acts.items():
                    n = vals.numel()
                    if n == B:
                        vals = vals.repeat_interleave(T)
                        n = vals.numel()
                    if n != B * T:
                        continue
                    if is_real.any():
                        self.sum_real[name] += vals[is_real].sum().item()
                        self.cnt_real[name] += int(is_real.sum().item())
                    if is_fake.any():
                        self.sum_fake[name] += vals[is_fake].sum().item()
                        self.cnt_fake[name] += int(is_fake.sum().item())

                self.batch_acts.clear()

        mean_real, mean_fake = {}, {}
        all_names = set(self.sum_real.keys()) | set(self.sum_fake.keys())
        for name in all_names:
            if self.cnt_real[name] > 0:
                mean_real[name] = self.sum_real[name] / self.cnt_real[name]
            if self.cnt_fake[name] > 0:
                mean_fake[name] = self.sum_fake[name] / self.cnt_fake[name]
        return mean_real, mean_fake

    def close(self):
        for h in self.hooks:
            h.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate mean activations")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="/outputs/")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick test runs")
    args = parser.parse_args()

    # Build model
    model = models.CAVMAESync(
        audio_length=1024,
        modality_specific_depth=11,
        num_register_tokens=4,
        total_frame=16,
        cls_token=True,
    )
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    sd = torch.load(args.model_path, map_location="cpu")
    _ = model.load_state_dict(sd, strict=False)

    # Dataset config
    val_audio_conf = {
        "num_mel_bins": 128,
        "target_length": 1024,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "mode": "eval",
        "mean": -5.081,
        "std": 4.4849,
        "noise": False,
        "im_res": 224,
    }

    print("Loading validation dataset from", args.csv_path)
    val_dataset_full = custom_dataloader.VideoDataset(
        csv_path=args.csv_path,
        audio_conf=val_audio_conf
    )

    if args.max_samples is not None:
        print(f"⚠️ Limiting dataset to {args.max_samples} samples for test run")
        random_indices = random.sample(range(len(val_dataset_full)), args.max_samples)
        val_dataset = Subset(val_dataset_full, random_indices)
    else:
        val_dataset = val_dataset_full

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False
    )

    os.makedirs(args.output, exist_ok=True)
    evaluator = ActivationEvaluator(model, val_loader)
    evaluator.evaluate(args.output)
    evaluator.close()

    # Save activation means
    # all_layers = sorted(set(real_means.keys()) | set(fake_means.keys()))
    # rows = []
    # for name in all_layers:
    #     mr = real_means.get(name, float("nan"))
    #     mf = fake_means.get(name, float("nan"))
    #     diff = mf - mr if (mr == mr and mf == mf) else float("nan")
    #     rows.append((name, mr, mf, diff))
    # df = pd.DataFrame(rows, columns=["layer", "mean_real", "mean_fake", "diff"])
    # activation_csv_path = os.path.join(args.output, "activations_means.csv")
    # df.to_csv(activation_csv_path, index=False)
    # print(f"Saved activation means to {activation_csv_path}")

    # Save embeddings
    # df_emb = pd.DataFrame(evaluator.forward_logs)
    # embed_csv_path = os.path.join(args.output, "forward_embeddings.csv")
    # df_emb.to_csv(embed_csv_path, index=False)
    # print(f"Saved forward_feat embeddings to {embed_csv_path}")
