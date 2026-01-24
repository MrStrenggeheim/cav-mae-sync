import argparse
import logging
import os
import time
from datetime import timedelta

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.dataloader_sync import AudiosetDataset, unsupervised_collate_fn
from src.fakesync_config import FakeSyncConfig
from src.models.cav_mae_sync import CAVMAE


class CAVMAEModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # Handle args being a dict (when loading from checkpoint) or Namespace
        if isinstance(args, dict):
            from argparse import Namespace

            args = Namespace(**args)

        self.model = CAVMAE(
            img_size=args.im_res,
            audio_length=args.audio_length,
            embed_dim=args.embed_dim,
            total_frame=args.total_frame,
            contrastive_heads=args.contrastive_heads,
            cls_token=args.cls_token,
            num_register_tokens=args.num_register_tokens,
        )

        # Profiling (accumulate over log_freq steps, then report)
        self._profile_data_time = 0.0
        self._profile_forward_time = 0.0
        self._profile_step_count = 0
        self._batch_start_time = None

        # DDP sync: max batches per epoch (set by DDPBatchSyncCallback)
        self.max_batches_per_epoch = None

    def on_train_batch_start(self, batch, batch_idx):
        import time

        if self._batch_start_time is not None:
            self._profile_data_time += time.perf_counter() - self._batch_start_time
        self._batch_start_time = time.perf_counter()

    def forward(self, fbanks, images, mode="unsupervised_train"):
        return self.model(
            fbanks,
            images,
            mask_ratio_a=self.hparams.mask_ratio_a,
            mask_ratio_v=self.hparams.mask_ratio_v,
            mae_loss_weight=self.hparams.mae_loss_weight,
            contrast_loss_weight=self.hparams.contrast_loss_weight,
            contrast_bidirect=self.hparams.contrast_bidirect,
            contrast_inter_weight=self.hparams.contrast_inter_weight,
            contrast_intra_weight=self.hparams.contrast_intra_weight,
            mode=mode,
        )

    def training_step(self, batch, batch_idx):
        import sys
        import time

        print(
            f"[Rank {self.global_rank}] Batch {batch_idx}: Training Step Start",
            flush=True,
        )

        # Note: DDP sync is handled by setting trainer.limit_train_batches in DDPBatchSyncCallback.
        # This ensures all ranks process exactly the same number of batches per epoch.
        # The max_batches_per_epoch attribute is now only used for informational logging.

        fbanks, images, video_ids, frame_indices = batch

        print(f"[Rank {self.global_rank}] Batch {batch_idx}: Forward Start", flush=True)
        forward_start = time.perf_counter()
        outputs = self(fbanks, images)
        forward_time = time.perf_counter() - forward_start
        print(
            f"[Rank {self.global_rank}] Batch {batch_idx}: Forward End ({forward_time:.3f}s)",
            flush=True,
        )
        self._profile_forward_time += forward_time
        self._profile_step_count += 1

        # Unpack outputs from dictionary
        loss = outputs["loss"]
        loss_mae = outputs["loss_mae"]
        loss_c = outputs["loss_c"]
        intra_acc = outputs["c_acc"]
        inter_acc = outputs["inter_acc"]

        cls_a = outputs.get("cls_a")
        cls_v = outputs.get("cls_v")
        if cls_a is not None and cls_v is not None:
            with torch.no_grad():
                # Handle case where outputs are patch embeddings [batch, patches, embed]
                # vs CLS tokens [batch, embed]
                if cls_a.dim() == 3:
                    cls_a = cls_a.mean(dim=1)
                if cls_v.dim() == 3:
                    cls_v = cls_v.mean(dim=1)

                cls_a_norm = torch.nn.functional.normalize(cls_a, dim=-1)
                cls_v_norm = torch.nn.functional.normalize(cls_v, dim=-1)

                per_sample_sim = (cls_a_norm * cls_v_norm).sum(dim=-1)

                agg_method = self.hparams.get("sync_aggregation", "all")

                if agg_method == "all":
                    # Log all aggregation methods
                    # 1. Epoch-level logging (always active for accurate global mean)
                    self.log(
                        "train/sync/mean",
                        per_sample_sim.mean(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        batch_size=len(video_ids),
                    )
                    self.log(
                        "train/sync/min",
                        per_sample_sim.min(),
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        batch_size=len(video_ids),
                    )
                    self.log(
                        "train/sync/p10",
                        torch.quantile(per_sample_sim, 0.1),
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        batch_size=len(video_ids),
                    )
                    self.log(
                        "train/sync/p25",
                        torch.quantile(per_sample_sim, 0.25),
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        batch_size=len(video_ids),
                    )
                    self.log(
                        "train/sync/variance",
                        per_sample_sim.var(),
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        batch_size=len(video_ids),
                    )

                    # 2. Step-level logging (sampled every log_freq steps for visibility)
                    log_freq = self.hparams.get("log_freq", 100)
                    if batch_idx % log_freq == 0:
                        self.log(
                            "train/sync/mean_step",
                            per_sample_sim.mean(),
                            on_step=True,
                            on_epoch=False,
                            logger=True,
                            batch_size=len(video_ids),
                        )
                        self.log(
                            "train/sync/min_step",
                            per_sample_sim.min(),
                            on_step=True,
                            on_epoch=False,
                            logger=True,
                            batch_size=len(video_ids),
                        )
                        self.log(
                            "train/sync/p10_step",
                            torch.quantile(per_sample_sim, 0.1),
                            on_step=True,
                            on_epoch=False,
                            logger=True,
                            batch_size=len(video_ids),
                        )
                        self.log(
                            "train/sync/p25_step",
                            torch.quantile(per_sample_sim, 0.25),
                            on_step=True,
                            on_epoch=False,
                            logger=True,
                            batch_size=len(video_ids),
                        )
                        self.log(
                            "train/sync/variance_step",
                            per_sample_sim.var(),
                            on_step=True,
                            on_epoch=False,
                            logger=True,
                            batch_size=len(video_ids),
                        )
                else:
                    # Log only the specified method
                    if agg_method == "mean":
                        sync_score = per_sample_sim.mean()
                    elif agg_method == "min":
                        sync_score = per_sample_sim.min()
                    elif agg_method == "p10":
                        sync_score = torch.quantile(per_sample_sim, 0.1)
                    elif agg_method == "p25":
                        sync_score = torch.quantile(per_sample_sim, 0.25)
                    else:
                        sync_score = per_sample_sim.mean()

                    # Epoch log
                    self.log(
                        "train/sync_score",
                        sync_score,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        batch_size=len(video_ids),
                    )
                    self.log(
                        "train/sync/variance",
                        per_sample_sim.var(),
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        batch_size=len(video_ids),
                    )

                    # Step log (sampled)
                    log_freq = self.hparams.get("log_freq", 100)
                    if batch_idx % log_freq == 0:
                        self.log(
                            "train/sync_score_step",
                            sync_score,
                            on_step=True,
                            on_epoch=False,
                            prog_bar=True,
                            logger=True,
                            batch_size=len(video_ids),
                        )
                        self.log(
                            "train/sync/variance_step",
                            per_sample_sim.var(),
                            on_step=True,
                            on_epoch=False,
                            logger=True,
                            batch_size=len(video_ids),
                        )

        # Logging
        self.log(
            "train/loss/total",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(video_ids),
        )
        self.log(
            "train_loss_total",
            loss,
            on_step=True,
            on_epoch=True,
            logger=False,
            batch_size=len(video_ids),
        )  # For checkpoint filename
        self.log(
            "train/loss/mae",
            loss_mae,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(video_ids),
        )
        self.log(
            "train/loss/contrast",
            loss_c,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=len(video_ids),
        )
        self.log(
            "train/acc/intra",
            intra_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(video_ids),
        )
        self.log(
            "train/acc/inter",
            inter_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(video_ids),
        )

        # Log learning rate manually to ensure it's in TensorBoard with a clear name
        # (LearningRateMonitor also logs it, but as lr-AdamW)
        optimizers = self.trainer.optimizers
        if optimizers:
            lr = optimizers[0].param_groups[0]["lr"]
            self.log(
                "train/lr", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True
            )

        # Profile logging every log_freq steps
        log_freq = self.hparams.get("log_freq", 100)
        if self._profile_step_count >= log_freq and self._profile_step_count > 0:
            avg_data_time = (
                self._profile_data_time / self._profile_step_count * 1000
            )  # ms
            avg_forward_time = (
                self._profile_forward_time / self._profile_step_count * 1000
            )  # ms

            self.log(
                "profile/batch_gap_ms",
                avg_data_time,
                on_step=True,
                logger=True,
                batch_size=len(video_ids),
            )
            self.log(
                "profile/forward_ms",
                avg_forward_time,
                on_step=True,
                logger=True,
                batch_size=len(video_ids),
            )

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
                logging.debug(
                    f"[Profile] batch_gap: {avg_data_time:.1f}ms, forward: {avg_forward_time:.1f}ms, "
                    f"GPU_mem: {gpu_mem:.1f}GB (avg over {self._profile_step_count} steps)"
                )
            else:
                logging.debug(
                    f"[Profile] batch_gap: {avg_data_time:.1f}ms, forward: {avg_forward_time:.1f}ms "
                    f"(avg over {self._profile_step_count} steps)"
                )

            # Reset counters
            self._profile_data_time = 0.0
            self._profile_forward_time = 0.0
            self._profile_step_count = 0

        print(
            f"[Rank {self.global_rank}] Batch {batch_idx}: Returning Loss", flush=True
        )
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        import time

        print(f"[Rank {self.global_rank}] Starting Batch {batch_idx}", flush=True)
        if self._batch_start_time is not None:
            self._profile_data_time += time.perf_counter() - self._batch_start_time
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        import time

        self._batch_start_time = time.perf_counter()
        print(f"[Rank {self.global_rank}] Finished Batch {batch_idx}", flush=True)

    def on_before_zero_grad(self, optimizer):
        print(
            f"[Rank {self.global_rank}] Optimizer Step (Zero Grad) - Start", flush=True
        )

    def on_before_backward(self, loss):
        print(
            f"[Rank {self.global_rank}] Backward Pass - Start (DDP Sync happens here)",
            flush=True,
        )

    def on_after_backward(self):
        print(f"[Rank {self.global_rank}] Backward Pass - End", flush=True)

    def on_before_optimizer_step(self, optimizer):
        print(f"[Rank {self.global_rank}] Optimizer Step - Start", flush=True)

    def validation_step(self, batch, batch_idx):
        # Alias for test_step
        return self.test_step(batch, batch_idx)

    def on_train_epoch_end(self):
        print(
            f"[Rank {self.global_rank}] Epoch End - Start (Checkpointing/Sync happens here)",
            flush=True,
        )

    def on_save_checkpoint(self, checkpoint):
        print(f"[Rank {self.global_rank}] Saving Checkpoint - Start", flush=True)

    def test_step(self, batch, batch_idx):
        print(
            f"[Rank {self.global_rank}] Test/Val Batch {batch_idx}: Start", flush=True
        )
        fbanks, images, video_ids, frame_indices = batch

        t0 = time.time()
        outputs = self(fbanks, images)
        t1 = time.time()
        print(
            f"[Rank {self.global_rank}] Test/Val Batch {batch_idx}: Forward done ({t1-t0:.3f}s)",
            flush=True,
        )

        # Unpack outputs from dictionary
        loss = outputs["loss"]
        loss_mae = outputs["loss_mae"]
        loss_c = outputs["loss_c"]
        intra_acc = outputs["c_acc"]
        inter_acc = outputs["inter_acc"]

        # Logging
        self.log(
            "test/loss/total",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("test/loss/mae", loss_mae, on_step=True, on_epoch=True, logger=True)
        self.log("test/loss/contrast", loss_c, on_step=True, on_epoch=True, logger=True)
        self.log(
            "test/acc/intra",
            intra_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test/acc/inter",
            inter_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        print(f"[Rank {self.global_rank}] Test/Val Batch {batch_idx}: End", flush=True)
        return loss

    def configure_optimizers(self):
        # Use fused AdamW for faster CUDA execution (PyTorch 2.0+)
        fused = torch.cuda.is_available() and hasattr(torch.optim.AdamW, "fused")
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            fused=fused,
        )

        if self.hparams.warmup_epochs > 0:
            # Minimum LR is 1% of base LR to prevent complete learning stagnation
            # This is especially important for longer training runs (e.g., 20 epochs)
            min_lr = self.hparams.lr * 0.01

            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=self.hparams.warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.epochs - self.hparams.warmup_epochs,
                eta_min=min_lr,
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.hparams.warmup_epochs],
            )

            logging.info(
                f"LR Scheduler: {self.hparams.warmup_epochs} warmup epochs, "
                f"{self.hparams.epochs - self.hparams.warmup_epochs} cosine decay epochs, "
                f"base_lr={self.hparams.lr}, min_lr={min_lr}"
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer


class CAVMAEDataModule(pl.LightningDataModule):
    """
    LightningDataModule for CAV-MAE training with proper DDP support.

    Uses Lightning's limit_train_batches to ensure all ranks process the same
    number of batches per epoch, preventing NCCL deadlocks.
    """

    def __init__(self, args, audio_conf):
        super().__init__()
        self.args = args
        self.audio_conf = audio_conf
        self.dataset = None
        self.min_batches_per_epoch = None  # Set during setup()
        self.discarded_samples_info = None  # For logging

    def setup(self, stage=None):
        """
        Called after DDP initialization on each process.
        Creates the dataset and computes synchronized batch count.
        """
        import torch.distributed as dist

        if self.args.sharded_dataset_dir:
            from src.dataloader_sharded import ShardedAudiosetDataset

            # Get rank and world_size from trainer (available after DDP init)
            if self.trainer and hasattr(self.trainer, "global_rank"):
                rank = self.trainer.global_rank
                world_size = self.trainer.world_size
            elif dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = int(os.environ.get("RANK", 0))
                world_size = int(os.environ.get("WORLD_SIZE", 1))

            logging.info(f"DataModule setup: rank={rank}, world_size={world_size}")

            # Create dataset with explicit rank/world_size (NO max_samples - we limit at Trainer level)
            self.dataset = ShardedAudiosetDataset(
                shard_dir=self.args.sharded_dataset_dir,
                audio_conf=self.audio_conf,
                use_mmap=self.args.use_mmap,
                dataset_fraction=self.args.dataset_fraction,
                rank=rank,
                world_size=world_size,
            )

            # Compute synchronized batch count for DDP
            if world_size > 1:
                local_samples = self.dataset.count_samples_for_rank()
                batch_size = self.args.batch_size
                local_batches = local_samples // batch_size

                logging.info(
                    f"Rank {rank}: {local_samples} samples -> {local_batches} batches (batch_size={batch_size})"
                )

                # Sync minimum batch count across all ranks
                batch_tensor = torch.tensor(
                    [local_batches],
                    dtype=torch.long,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                dist.all_reduce(batch_tensor, op=dist.ReduceOp.MIN)
                self.min_batches_per_epoch = batch_tensor.item()

                # Log discarded samples for monitoring
                used_samples = self.min_batches_per_epoch * batch_size
                discarded = local_samples - used_samples
                pct = (discarded / local_samples * 100) if local_samples > 0 else 0

                self.discarded_samples_info = {
                    "rank": rank,
                    "local_samples": local_samples,
                    "used_samples": used_samples,
                    "discarded": discarded,
                    "discard_pct": pct,
                }

                logging.info(
                    f"Rank {rank}: DDP sync -> {self.min_batches_per_epoch} batches/epoch "
                    f"(discarding {discarded} samples = {pct:.1f}%)"
                )

                # Enforce per-worker sample limiting to prevent DataLoader hangs
                self.dataset.set_max_samples(used_samples)
            else:
                # Single GPU: use all available batches
                local_samples = self.dataset.count_samples_for_rank()
                self.min_batches_per_epoch = local_samples // self.args.batch_size
                logging.info(f"Single GPU: {self.min_batches_per_epoch} batches/epoch")
        else:
            # Non-sharded dataset (legacy JSON-based)
            self.dataset = AudiosetDataset(
                dataset_json_file=self.args.dataset_json,
                audio_conf=self.audio_conf,
                label_csv=self.args.label_csv,
            )
            self.min_batches_per_epoch = None  # Let DataLoader handle it

    def train_dataloader(self):
        if self.args.sharded_dataset_dir:
            return DataLoader(
                self.dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=unsupervised_collate_fn,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True if self.args.num_workers > 0 else False,
                prefetch_factor=4 if self.args.num_workers > 0 else None,
                timeout=(
                    120 if self.args.num_workers > 0 else 0
                ),  # 2 minute timeout for workers
            )
        else:
            return DataLoader(
                self.dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=unsupervised_collate_fn,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True if self.args.num_workers > 0 else False,
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Train CAV-MAE Sync Unsupervised (Lightning)"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_json", type=str, default=None, help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--sharded_dataset_dir",
        type=str,
        default=None,
        help="Path to sharded dataset directory (overrides dataset_json)",
    )
    parser.add_argument(
        "--label_csv", type=str, default=None, help="Path to label CSV file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (number of videos)"
    )
    # Try to determine safe default workers respecting affinity (Slurm friendly)
    # Try to determine safe default workers respecting affinity and multi-GPU
    try:
        # Total CPUs assigned to this task/job
        if os.environ.get("SLURM_CPUS_PER_TASK"):
            total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK"))
        else:
            total_cpus = len(os.sched_getaffinity(0))

        # If running on multiple GPUs on this node, divide workers
        ngpus = torch.cuda.device_count()
        if ngpus > 1:
            default_workers = max(1, total_cpus // ngpus)
            logging.info(
                f"Auto-detected {total_cpus} CPUs and {ngpus} GPUs. Setting default workers to {default_workers} per GPU."
            )
        else:
            default_workers = total_cpus

    except (AttributeError, NotImplementedError, OSError):
        default_workers = os.cpu_count() or 4

    parser.add_argument(
        "--num_workers",
        type=int,
        default=default_workers,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--total_frame", type=int, default=16, help="Number of frames per video"
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=48,
        help="Target audio length (must be divisible by 16)",
    )
    parser.add_argument("--im_res", type=int, default=224, help="Image resolution")

    # Model arguments
    parser.add_argument(
        "--audio_length",
        type=int,
        default=48,
        help="Audio length for model (must match target_length, divisible by 16)",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument(
        "--mask_ratio_a", type=float, default=0.75, help="Audio mask ratio"
    )
    parser.add_argument(
        "--mask_ratio_v", type=float, default=0.75, help="Video mask ratio"
    )
    parser.add_argument(
        "--mae_loss_weight", type=float, default=1.0, help="Weight for MAE loss"
    )
    parser.add_argument(
        "--contrast_loss_weight",
        type=float,
        default=0.01,
        help="Weight for contrastive loss",
    )
    parser.add_argument(
        "--contrast_bidirect",
        action="store_true",
        help="Use bidirectional contrastive loss",
    )
    parser.add_argument(
        "--contrast_inter_weight",
        type=float,
        default=0.4,
        help="Weight for inter-sample contrastive loss",
    )
    parser.add_argument(
        "--contrast_intra_weight",
        type=float,
        default=0.6,
        help="Weight for intra-sample contrastive loss",
    )
    parser.add_argument("--cls_token", action="store_true", help="Use CLS token")
    parser.add_argument(
        "--num_register_tokens", type=int, default=8, help="Number of register tokens"
    )
    parser.add_argument(
        "--contrastive_heads", action="store_true", help="Use contrastive heads"
    )
    parser.add_argument(
        "--sync_aggregation",
        type=str,
        default="all",
        choices=["all", "mean", "min", "p10", "p25"],
        help="Aggregation method for sync score ('all' logs all methods)",
    )

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--warmup_epochs", type=int, default=1, help="Number of warmup epochs"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--log_freq", type=int, default=10, help="Logging frequency (steps)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to resume checkpoint (ckpt file)"
    )
    parser.add_argument(
        "--fast_dev_run", action="store_true", help="Run a quick development run"
    )
    parser.add_argument(
        "--checkpoint_interval_hours",
        type=float,
        default=1.0,
        help="Save checkpoint every N hours",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (default: True)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps (simulate larger batch)",
    )

    # Audio Conf defaults
    parser.add_argument(
        "--num_mel_bins", type=int, default=128, help="Number of mel bins"
    )
    parser.add_argument("--mean", type=float, default=-6.166528, help="Dataset mean")
    parser.add_argument("--std", type=float, default=3.483568, help="Dataset std")

    # Dataloader robustness options
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        help="Use memory-mapped I/O for shard loading. Only enable for local SSD/NVMe. "
        "Disabled by default for network storage (NFS/Lustre) compatibility.",
    )
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.0-1.0). Useful for quick testing. Default: 1.0 (full)",
    )

    return parser.parse_args()


def main():
    # Configure logging
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level, format="%(asctime)s [%(levelname)s] %(message)s", force=True
    )

    # Suppress verbose third-party logs
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Set float32 matmul precision to 'high'/'medium' (instead of 'highest') to utilize Tensor Cores if available
    torch.set_float32_matmul_precision("medium")

    # Enable Flash Attention if available (PyTorch 2.0+)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logging.info("Flash Attention enabled")

    # Configure logging
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
    args = get_args()

    # Validate configuration (will raise ValueError if invalid)
    config = FakeSyncConfig.from_args(args)
    logging.info("Configuration validated successfully")

    # Dataset Configuration (generated from validated config)
    audio_conf = config.to_audio_conf()
    logging.info("Audio Configuration:")
    logging.info(audio_conf)

    # Create DataModule - dataset creation happens in setup() after DDP initialization
    # This is critical for proper DDP synchronization with IterableDataset
    data_module = CAVMAEDataModule(args, audio_conf)

    model = CAVMAEModule(args)

    if getattr(args, "gradient_checkpointing", False):
        model.model.enable_gradient_checkpointing()
        logging.info("Gradient checkpointing enabled")

    # NOTE: torch.compile disabled - CAV-MAE uses dynamic shapes in forward_decoder
    # (e.g., int(mask_a[0].sum())) which causes constant graph breaks and recompilations.
    # if hasattr(torch, 'compile'):
    #     logging.info("Applying torch.compile to model...")
    #     model = torch.compile(model, mode='reduce-overhead')

    # Checkpoint callback for top-k best models (saved at epoch end based on monitored metric)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_path,
        filename="cav-mae-{epoch:02d}-{train_loss_total:.2f}",
        save_top_k=3,
        save_last=True,  # Also save last.ckpt at each epoch end
        monitor="train/loss/total",
        mode="min",
        every_n_epochs=1,  # Ensure we save at every epoch boundary
    )

    # Separate callback for periodic time-based checkpoints (disaster recovery)
    # Note: With IterableDataset, mid-epoch checkpoints will restart at epoch beginning when resumed
    time_checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_path,
        filename="cav-mae-time-{epoch:02d}-step{step}",
        save_top_k=-1,  # Keep all time-based checkpoints
        train_time_interval=timedelta(hours=args.checkpoint_interval_hours),
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # TensorBoard Logger
    tb_logger = TensorBoardLogger(
        save_dir=args.save_path, name="tensorboard", version=None
    )

    if torch.cuda.is_available():
        # DDP strategy: find_unused_parameters=True required because model has
        # conditional parameter usage (e.g., contrastive heads, different return paths)
        strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False,
        )
    else:
        strategy = "auto"

    # Determine precision: use bf16 on Ampere+ GPUs (compute >= 8.0), else fp16
    precision = 32  # Default for CPU
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] >= 8:  # Ampere or newer
            precision = "bf16-mixed"
            logging.info(
                f"Using bfloat16 precision (GPU compute capability {device_capability[0]}.{device_capability[1]} >= 8.0)"
            )
        else:
            precision = "16-mixed"
            logging.info(
                f"Using float16 precision (GPU compute capability {device_capability[0]}.{device_capability[1]} < 8.0)"
            )
    # Note: We cannot call data_module.setup() here because DDP is not initialized yet.
    # Lightning calls setup() after DDP init. The min_batches_per_epoch is computed there
    # and we need to update limit_train_batches dynamically.
    #
    # SOLUTION: Use a Callback to update trainer.limit_train_batches after setup() runs.

    class DDPBatchSyncCallback(pl.Callback):
        """Callback to sync batch counts across DDP ranks after datamodule setup.

        Sets trainer.limit_train_batches to ensure all ranks process the same number
        of batches per epoch, preventing NCCL deadlocks with IterableDataset.

        Also updates the progress bar total so tqdm displays properly.
        """

        def on_train_start(self, trainer, pl_module):
            """Called after setup() and before training starts."""
            if hasattr(trainer.datamodule, "min_batches_per_epoch"):
                min_batches = trainer.datamodule.min_batches_per_epoch
                if min_batches is not None:
                    # Set trainer's limit - this controls when the epoch ends
                    trainer.limit_train_batches = min_batches

                    # Also set on the module for informational purposes
                    if hasattr(pl_module, "max_batches_per_epoch"):
                        pl_module.max_batches_per_epoch = min_batches

                    logging.info(
                        f"DDPBatchSyncCallback: Synced to {min_batches} batches/epoch (progress bar will show total)"
                    )

                    # Log discarded samples info if available
                    if trainer.datamodule.discarded_samples_info:
                        info = trainer.datamodule.discarded_samples_info
                        logging.info(
                            f"Rank {info['rank']}: Using {info['used_samples']} samples, "
                            f"discarding {info['discarded']} ({info['discard_pct']:.1f}%)"
                        )

        def on_train_epoch_start(self, trainer, pl_module):
            """Manually update the progress bar total since num_training_batches is read-only."""
            if hasattr(trainer.datamodule, "min_batches_per_epoch"):
                min_batches = trainer.datamodule.min_batches_per_epoch
                if min_batches is not None:
                    # Try to access the progress bar directly
                    # Standard TQDMProgressBar (PL default) stores the actual tqdm instance in .main_progress_bar
                    if trainer.progress_bar_callback:
                        pbar = getattr(
                            trainer.progress_bar_callback, "main_progress_bar", None
                        )
                        if pbar is not None:
                            pbar.total = min_batches
                            pbar.refresh()

                        trainer.progress_bar_callback.main_progress_bar.total = (
                            min_batches
                        )
                        trainer.progress_bar_callback.main_progress_bar.refresh()

    # Add to callbacks
    ddp_sync_callback = DDPBatchSyncCallback()

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy=strategy,
        precision=precision,
        callbacks=[
            checkpoint_callback,
            time_checkpoint_callback,
            lr_monitor,
            ddp_sync_callback,
        ],
        logger=tb_logger,
        log_every_n_steps=args.log_freq,
        default_root_dir=args.save_path,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )

    # Log the exact path where TensorBoard events will be written
    if trainer.is_global_zero:
        logging.info(
            f"TensorBoard Logger initialized. Logs will be written to: {tb_logger.log_dir}"
        )

    ckpt_path = args.resume
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Checking checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location="cpu")
            # Heuristic to detect PL checkpoint
            if "pytorch-lightning_version" in checkpoint or "callbacks" in checkpoint:
                logging.info(
                    "Detected PyTorch Lightning checkpoint. Resuming training state..."
                )
                ckpt_path = args.resume
            else:
                logging.info(
                    "Detected raw PyTorch checkpoint. Loading model weights only..."
                )
                state_dict = checkpoint.get(
                    "model", checkpoint.get("state_dict", checkpoint)
                )

                # Remove 'module.' prefix if present
                clean_state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }

                # Determine target model and key prefix
                if any(k.startswith("model.") for k in clean_state_dict):
                    target_model = model
                    key_prefix = "model."
                else:
                    target_model = model.model
                    key_prefix = ""

                # Audio-length dependent parameters that may have shape mismatch
                # These depend on num_patches_a = int((audio_length / 16) * (128 / 16))
                audio_length_dependent_keys = [
                    "pos_embed_a",
                    "decoder_pos_embed_a",
                ]

                # Check for shape mismatches and filter out incompatible weights
                filtered_state_dict = {}
                skipped_keys = []

                for key, value in clean_state_dict.items():
                    # Get the corresponding key in target model
                    model_key = (
                        key.replace(key_prefix, "")
                        if key_prefix and key.startswith(key_prefix)
                        else key
                    )
                    full_key = f"{key_prefix}{model_key}" if key_prefix else model_key

                    # Check if this is an audio-length dependent key
                    is_audio_dependent = any(
                        dep_key in key for dep_key in audio_length_dependent_keys
                    )

                    if is_audio_dependent:
                        # Get expected shape from current model
                        try:
                            if key_prefix:
                                expected_param = target_model.state_dict().get(key)
                            else:
                                expected_param = target_model.state_dict().get(key)

                            if (
                                expected_param is not None
                                and expected_param.shape != value.shape
                            ):
                                logging.warning(
                                    f"Shape mismatch for '{key}': "
                                    f"checkpoint has {list(value.shape)}, "
                                    f"model expects {list(expected_param.shape)}. "
                                    f"This weight will be randomly initialized."
                                )
                                skipped_keys.append(key)
                                continue
                        except Exception:
                            pass

                    filtered_state_dict[key] = value

                if skipped_keys:
                    logging.warning(
                        f"Skipped {len(skipped_keys)} audio-length dependent weights due to shape mismatch: "
                        f"{skipped_keys}. These will be reinitialized with sin-cos positional encoding."
                    )

                # Load filtered weights
                if key_prefix:
                    msg = target_model.load_state_dict(
                        filtered_state_dict, strict=False
                    )
                else:
                    msg = target_model.load_state_dict(
                        filtered_state_dict, strict=False
                    )

                logging.info(f"Weights loaded manually. Message: {msg}")

                # Reinitialize skipped positional embeddings with sin-cos encoding
                if skipped_keys:
                    import numpy as np

                    from src.models.pos_embed import get_2d_sincos_pos_embed

                    # Get the actual CAVMAE model (unwrap from Lightning module if needed)
                    cavmae_model = (
                        target_model.model
                        if hasattr(target_model, "model")
                        else target_model
                    )

                    for key in skipped_keys:
                        if "pos_embed_a" in key and "decoder" not in key:
                            # Encoder audio positional embedding
                            embed_dim = cavmae_model.pos_embed_a.shape[-1]
                            num_patches = cavmae_model.patch_embed_a.num_patches
                            pos_embed = get_2d_sincos_pos_embed(
                                embed_dim,
                                8,  # frequency dimension (128 / 16)
                                int(num_patches / 8),  # time dimension
                                cls_token=False,
                            )
                            cavmae_model.pos_embed_a.data.copy_(
                                torch.from_numpy(pos_embed).float().unsqueeze(0)
                            )
                            logging.info(
                                f"Reinitialized '{key}' with sin-cos positional encoding (shape: {cavmae_model.pos_embed_a.shape})"
                            )

                        elif "decoder_pos_embed_a" in key:
                            # Decoder audio positional embedding
                            embed_dim = cavmae_model.decoder_pos_embed_a.shape[-1]
                            num_patches = cavmae_model.patch_embed_a.num_patches
                            pos_embed = get_2d_sincos_pos_embed(
                                embed_dim,
                                8,  # frequency dimension
                                int(num_patches / 8),  # time dimension
                                cls_token=False,
                            )
                            cavmae_model.decoder_pos_embed_a.data.copy_(
                                torch.from_numpy(pos_embed).float().unsqueeze(0)
                            )
                            logging.info(
                                f"Reinitialized '{key}' with sin-cos positional encoding (shape: {cavmae_model.decoder_pos_embed_a.shape})"
                            )

                ckpt_path = None  # Do not resume trainer state

        except Exception as e:
            logging.error(f"Failed to inspect checkpoint {args.resume}. Error: {e}")
            raise

    logging.info("Starting training...")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
