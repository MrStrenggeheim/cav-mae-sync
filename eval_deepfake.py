"""
Deepfake Detection Evaluation Script for CAV-MAE Sync

Evaluates audio-visual synchronization quality to detect deepfakes.
Computes per-frame sync scores (cosine similarity between audio and visual embeddings)
and aggregates them using multiple strategies for robust deepfake detection.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.dataloader_sync import AudiosetDataset, unsupervised_collate_fn
from src.dataloader_sharded import ShardedAudiosetDataset
from train_unsupervised import CAVMAEModule

import logging

# =============================================================================
# Sync Score Computation Utilities
# =============================================================================

@dataclass
class SyncScoreResult:
    """Container for sync score results from a single video."""
    video_id: str
    per_frame_similarities: np.ndarray  # Shape: (num_frames,)
    
    # Aggregated scores
    sync_mean: float
    sync_min: float
    sync_max: float
    sync_std: float
    sync_p10: float  # 10th percentile (robust worst-case)
    sync_p25: float  # 25th percentile
    sync_p50: float  # Median
    
    # Optional: loss metrics
    loss: float = 0.0
    loss_mae: float = 0.0
    loss_c: float = 0.0
    intra_acc: float = 0.0
    inter_acc: float = 0.0
    
    # Extended metrics
    intra_sim_audio: float = 0.0   # Audio temporal coherence
    intra_sim_visual: float = 0.0  # Visual temporal coherence
    sync_euc: float = 0.0          # Euclidean distance
    sync_pearson: float = 0.0      # Pearson correlation


def compute_sync_scores(
    cls_a: torch.Tensor, 
    cls_v: torch.Tensor,
    total_frames: int = 16
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute per-frame synchronization scores between audio and visual embeddings.
    
    Args:
        cls_a: Audio CLS tokens, shape (batch_size * total_frames, embed_dim)
        cls_v: Visual CLS tokens, shape (batch_size * total_frames, embed_dim)
        total_frames: Number of frames per video
        
    Returns:
        per_frame_sims: Per-frame cosine similarities, shape (batch_size, total_frames)
        aggregated: Dict of aggregated scores per video, each shape (batch_size,)
    """
    # Normalize embeddings
    cls_a_norm = F.normalize(cls_a, p=2, dim=-1)
    cls_v_norm = F.normalize(cls_v, p=2, dim=-1)
    
    # Compute per-frame cosine similarity
    # Element-wise product followed by sum gives cosine similarity for normalized vectors
    per_frame_sims = (cls_a_norm * cls_v_norm).sum(dim=-1)  # (batch_size * total_frames,)
    
    # Reshape to (batch_size, total_frames)
    n_samples = per_frame_sims.shape[0]
    if n_samples % total_frames != 0:
        raise ValueError(
            f"Number of samples ({n_samples}) is not divisible by total_frames ({total_frames}). "
            f"This suggests a batch size issue."
        )
    
    batch_size = n_samples // total_frames
    per_frame_sims = per_frame_sims.view(batch_size, total_frames)
    
    # Compute aggregated metrics per video
    aggregated = {
        'sync_mean': per_frame_sims.mean(dim=1),
        'sync_min': per_frame_sims.min(dim=1).values,
        'sync_max': per_frame_sims.max(dim=1).values,
        'sync_std': per_frame_sims.std(dim=1),
    }
    
    # Percentiles (need to move to CPU for numpy operations, then back to tensor)
    sims_np = per_frame_sims.cpu().numpy()
    aggregated['sync_p10'] = torch.tensor(np.percentile(sims_np, 10, axis=1), device=per_frame_sims.device)
    aggregated['sync_p25'] = torch.tensor(np.percentile(sims_np, 25, axis=1), device=per_frame_sims.device)
    aggregated['sync_p50'] = torch.tensor(np.percentile(sims_np, 50, axis=1), device=per_frame_sims.device)
    
    # --- New Metrics ---
    # Reshape embeddings to (batch_size, total_frames, embed_dim)
    cls_a_batches = cls_a_norm.view(batch_size, total_frames, -1)
    cls_v_batches = cls_v_norm.view(batch_size, total_frames, -1)
    
    # 1. Intra-Modality Similarity (Temporal Coherence)
    # Compute mean cosine sim of all frame pairs within the same video
    # (b, t, d) @ (b, d, t) -> (b, t, t)
    sim_aa = torch.bmm(cls_a_batches, cls_a_batches.transpose(1, 2))
    sim_vv = torch.bmm(cls_v_batches, cls_v_batches.transpose(1, 2))
    
    # Exclude diagonal (self-similarity is always 1)
    mask = ~torch.eye(total_frames, device=cls_a.device, dtype=torch.bool)
    aggregated['intra_sim_audio'] = (sim_aa * mask).sum(dim=(1, 2)) / (total_frames * (total_frames - 1))
    aggregated['intra_sim_visual'] = (sim_vv * mask).sum(dim=(1, 2)) / (total_frames * (total_frames - 1))

    # 2. Euclidean Distance (on normalized embeddings)
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2u.v = 2 - 2cos(u,v) for normalized vectors
    # But let's compute directly for safety
    dist = torch.norm(cls_a_norm - cls_v_norm, dim=-1).view(batch_size, total_frames)
    aggregated['sync_euc'] = dist.mean(dim=1)

    # 3. Pearson Correlation (between feature vectors, averaged over frames)
    # Pearson = cov(a, v) / (std(a)*std(v))
    # Note: cls_a_norm is L2 normalized, NOT centered/std-normalized.
    # We need to center and normalize by std dev for Pearson.
    
    # Center vectors
    cls_a_centered = cls_a - cls_a.mean(dim=-1, keepdim=True)
    cls_v_centered = cls_v - cls_v.mean(dim=-1, keepdim=True)
    
    # Pearson calculation
    pearson = F.cosine_similarity(cls_a_centered, cls_v_centered, dim=-1)
    aggregated['sync_pearson'] = pearson.view(batch_size, total_frames).mean(dim=1)
    
    return per_frame_sims, aggregated


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.
    
    Args:
        y_true: Ground truth labels (0=real, 1=fake)
        y_scores: Predicted scores (higher = more likely fake)
        
    Returns:
        eer: Equal Error Rate
        threshold: Threshold at EER
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the point where FPR and FNR are closest
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    threshold = thresholds[eer_idx]
    
    return eer, threshold


# =============================================================================
# Lightning Module for Evaluation
# =============================================================================

class DeepFakeEvaluator(pl.LightningModule):
    """
    Lightning module for evaluating deepfake detection using sync scores.
    
    Computes comprehensive sync score metrics and supports optional label-based
    evaluation with AUC-ROC and EER metrics.
    """
    
    def __init__(
        self, 
        model_module: CAVMAEModule, 
        use_masking: bool = False,
        total_frames: int = 16,
        labels_df: Optional[pd.DataFrame] = None
    ):
        """
        Args:
            model_module: Trained CAVMAEModule
            use_masking: Whether to use MAE masking during evaluation (default: False for cleaner embeddings)
            total_frames: Number of frames per video
            labels_df: Optional DataFrame with 'video_id' and 'label' columns (0=real, 1=fake)
        """
        super().__init__()
        self.model_module = model_module
        self.use_masking = use_masking
        self.total_frames = total_frames
        self.labels_df = labels_df
        
        # Storage for results
        self.test_outputs: List[SyncScoreResult] = []
        
    def forward(self, fbanks: torch.Tensor, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        if self.use_masking:
            # Use default mask ratios from checkpoint
            return self.model_module(fbanks, images)
        else:
            # No masking - use full context for cleaner embeddings
            return self.model_module.model(
                fbanks, images,
                mask_ratio_a=0.0,
                mask_ratio_v=0.0,
                mae_loss_weight=self.model_module.hparams.mae_loss_weight,
                contrast_loss_weight=self.model_module.hparams.contrast_loss_weight,
                mode='unsupervised_train'
            )

    def test_step(self, batch, batch_idx: int) -> SyncScoreResult:
        """Process a single video and compute sync scores."""
        fbanks, images, video_ids, frame_indices = batch
        
        # Forward pass
        with torch.no_grad():
            outputs = self(fbanks, images)
        
        # Extract CLS tokens
        cls_a = outputs['cls_a']  # (batch_size * total_frames, embed_dim)
        cls_v = outputs['cls_v']  # (batch_size * total_frames, embed_dim)
        
        # Compute sync scores
        per_frame_sims, aggregated = compute_sync_scores(
            cls_a, cls_v, self.total_frames
        )
        
        # For batch_size=1, we have a single video per step
        # Handle potential multi-video batches by iterating
        batch_size = per_frame_sims.shape[0]
        
        for i in range(batch_size):
            # video_ids is expanded per-frame [vid1, vid1, ..., vid2, vid2, ...]
            # We take the ID corresponding to the start of the i-th video
            idx = i * self.total_frames
            vid = video_ids[idx] if idx < len(video_ids) else f"unknown_{batch_idx}_{i}"
            
            result = SyncScoreResult(
                video_id=vid,
                per_frame_similarities=per_frame_sims[i].cpu().numpy(),
                sync_mean=aggregated['sync_mean'][i].item(),
                sync_min=aggregated['sync_min'][i].item(),
                sync_max=aggregated['sync_max'][i].item(),
                sync_std=aggregated['sync_std'][i].item(),
                sync_p10=aggregated['sync_p10'][i].item(),
                sync_p25=aggregated['sync_p25'][i].item(),
                sync_p50=aggregated['sync_p50'][i].item(),
                loss=outputs['loss'].item(),
                loss_mae=outputs['loss_mae'].item(),
                loss_c=outputs['loss_c'].item(),
                intra_acc=outputs['c_acc'].item(),
                inter_acc=outputs['inter_acc'].item(),
                # Extended metrics
                intra_sim_audio=aggregated['intra_sim_audio'][i].item(),
                intra_sim_visual=aggregated['intra_sim_visual'][i].item(),
                sync_euc=aggregated['sync_euc'][i].item(),
                sync_pearson=aggregated['sync_pearson'][i].item(),
            )
            self.test_outputs.append(result)
        
        # Log aggregated metrics (for first video in batch for logging)
        self.log('sync_mean', aggregated['sync_mean'][0], on_step=True, on_epoch=True, prog_bar=True)
        self.log('sync_min', aggregated['sync_min'][0], on_step=True, on_epoch=True)
        self.log('sync_std', aggregated['sync_std'][0], on_step=True, on_epoch=True)
        self.log('eval_loss', outputs['loss'], on_step=True, on_epoch=True)
        
        return result
    
    def on_test_epoch_end(self) -> None:
        """Compute final metrics and optionally AUC-ROC/EER if labels available."""
        if len(self.test_outputs) == 0:
            logging.warning("No test outputs collected!")
            return
            
        # Convert results to arrays
        sync_means = np.array([r.sync_mean for r in self.test_outputs])
        sync_mins = np.array([r.sync_min for r in self.test_outputs])
        sync_stds = np.array([r.sync_std for r in self.test_outputs])
        
        # Log summary statistics
        logging.info("\n" + "="*60)
        logging.info("SYNC SCORE SUMMARY")
        logging.info("="*60)
        logging.info(f"Total videos evaluated: {len(self.test_outputs)}")
        logging.info(f"Sync Mean - avg: {sync_means.mean():.4f}, std: {sync_means.std():.4f}")
        logging.info(f"Sync Min  - avg: {sync_mins.mean():.4f}, std: {sync_mins.std():.4f}")
        logging.info(f"Sync Std  - avg: {sync_stds.mean():.4f}, std: {sync_stds.std():.4f}")
        
        # Compute AUC-ROC and EER if labels are available
        if self.labels_df is not None:
            self._compute_detection_metrics()
    
    def _compute_detection_metrics(self) -> None:
        """Compute AUC-ROC and EER using provided labels."""
        # Create a dictionary for O(1) label lookup
        label_map = dict(zip(self.labels_df['video_id'], self.labels_df['label']))
        
        # Match labels to video IDs
        labels = []
        scores_by_metric = {
            'sync_mean': [],
            'sync_min': [],
            'sync_p10': [],
            'sync_p25': [],
            'sync_std': [],
            'sync_pearson': [],
            'sync_euc': [],
            'intra_sim_audio': [],
            'intra_sim_visual': [],
        }
        
        for result in self.test_outputs:
            vid = result.video_id
            if vid in label_map:
                label = label_map[vid]
                labels.append(label)
                # For sync scores, lower = more likely fake, so negate for sklearn
                scores_by_metric['sync_mean'].append(-result.sync_mean)
                scores_by_metric['sync_min'].append(-result.sync_min)
                scores_by_metric['sync_p10'].append(-result.sync_p10)
                scores_by_metric['sync_p25'].append(-result.sync_p25)
                scores_by_metric['sync_pearson'].append(-result.sync_pearson)
                # For distance/variance, higher = more likely fake
                scores_by_metric['sync_euc'].append(result.sync_euc)
                scores_by_metric['sync_std'].append(result.sync_std)
                # Intra-sim: Lower coherence = more likely fake? (we'll check AUC)
                scores_by_metric['intra_sim_audio'].append(-result.intra_sim_audio)
                scores_by_metric['intra_sim_visual'].append(-result.intra_sim_visual)
        
        if len(labels) < 2:
            logging.warning(f"Only {len(labels)} labels matched. Cannot compute detection metrics.")
            return
            
        labels = np.array(labels)
        
        # Check for class imbalance
        n_real = (labels == 0).sum()
        n_fake = (labels == 1).sum()
        if n_real == 0 or n_fake == 0:
            logging.warning(f"Labels are imbalanced (real={n_real}, fake={n_fake}). Cannot compute AUC.")
            return
        
        logging.info("\n" + "="*60)
        logging.info("DEEPFAKE DETECTION METRICS")
        logging.info("="*60)
        logging.info(f"Matched labels: {len(labels)} (real={n_real}, fake={n_fake})")
        
        for metric_name, scores in scores_by_metric.items():
            scores = np.array(scores)
            try:
                auc = roc_auc_score(labels, scores)
                eer, threshold = compute_eer(labels, scores)
                
                logging.info(f"\n{metric_name}:")
                logging.info(f"  AUC-ROC: {auc:.4f}")
                logging.info(f"  EER: {eer:.4f} (threshold: {threshold:.4f})")
                
                # Log to TensorBoard
                self.log(f'auc_{metric_name}', auc)
                self.log(f'eer_{metric_name}', eer)
                
            except ValueError as e:
                logging.warning(f"Could not compute metrics for {metric_name}: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CAV-MAE Sync for DeepFake Detection via Sync Scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic evaluation (no labels, just sync scores)
  python eval_deepfake.py \\
      --sharded_dataset_dir /path/to/shards \\
      --checkpoint_path outputs/checkpoints/latest.ckpt \\
      --output_csv results.csv

  # With labels for AUC-ROC computation
  python eval_deepfake.py \\
      --sharded_dataset_dir /path/to/shards \\
      --checkpoint_path outputs/checkpoints/latest.ckpt \\
      --label_csv labels.csv \\
      --output_csv results.csv
        """
    )
    
    # Dataset arguments
    parser.add_argument("--dataset_json", type=str, default=None, 
                        help="Path to dataset JSON file (legacy dataloader)")
    parser.add_argument("--sharded_dataset_dir", type=str, default=None, 
                        help="Path to sharded dataset directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to trained model checkpoint")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="Optional CSV with video_id,label columns (0=real, 1=fake)")
    parser.add_argument("--output_csv", type=str, default="eval_sync_results.csv", 
                        help="Path to save results CSV")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs", 
                        help="Output directory for logs and TensorBoard")
    
    # Dataloader arguments
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size (default: 1 for per-video evaluation)")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loading workers")
    parser.add_argument("--total_frame", type=int, default=16, 
                        help="Number of frames per video")
    parser.add_argument("--target_length", type=int, default=48, 
                        help="Target audio length (should match training)")
    parser.add_argument("--im_res", type=int, default=224, 
                        help="Image resolution")
    
    # Audio normalization (should match training)
    parser.add_argument("--num_mel_bins", type=int, default=128, 
                        help="Number of mel frequency bins")
    parser.add_argument("--mean", type=float, default=-6.166528, 
                        help="Dataset mean for audio normalization (VoxCeleb2 default)")
    parser.add_argument("--std", type=float, default=3.483568, 
                        help="Dataset std for audio normalization (VoxCeleb2 default)")
    
    # Evaluation config
    parser.add_argument("--use_masking", action="store_true", 
                        help="Enable MAE masking during evaluation (not recommended)")
    parser.add_argument("--no_masking", dest="use_masking", action="store_false",
                        help="Disable masking for cleaner embeddings (default)")
    parser.set_defaults(use_masking=False)  # Default to no masking
    
    return parser.parse_args()


def load_labels(label_csv: Optional[str]) -> Optional[pd.DataFrame]:
    """Load labels from CSV file if provided."""
    if label_csv is None:
        return None
        
    if not os.path.exists(label_csv):
        logging.warning(f"Label file not found: {label_csv}")
        return None
        
    df = pd.read_csv(label_csv)
    
    # Validate columns
    required_cols = {'video_id', 'label'}
    if not required_cols.issubset(set(df.columns)):
        logging.warning(f"Label CSV missing required columns: {required_cols - set(df.columns)}")
        return None
        
    # Validate labels
    unique_labels = set(df['label'].unique())
    if not unique_labels.issubset({0, 1}):
        logging.warning(f"Labels should be 0 (real) or 1 (fake), found: {unique_labels}")
        
    logging.info(f"Loaded {len(df)} labels from {label_csv}")
    return df


def results_to_dataframe(results: List[SyncScoreResult]) -> pd.DataFrame:
    """Convert list of SyncScoreResult to DataFrame."""
    data = []
    for r in results:
        row = {
            'video_id': r.video_id,
            'sync_mean': r.sync_mean,
            'sync_min': r.sync_min,
            'sync_max': r.sync_max,
            'sync_std': r.sync_std,
            'sync_p10': r.sync_p10,
            'sync_p25': r.sync_p25,
            'sync_p50': r.sync_p50,
            'loss': r.loss,
            'loss_mae': r.loss_mae,
            'loss_c': r.loss_c,
            'intra_acc': r.intra_acc,
            'inter_acc': r.inter_acc,
            'intra_sim_audio': r.intra_sim_audio,
            'intra_sim_visual': r.intra_sim_visual,
            'sync_euc': r.sync_euc,
            'sync_pearson': r.sync_pearson,
        }
        # Also store per-frame similarities as a JSON-serializable list
        row['per_frame_sims'] = r.per_frame_similarities.tolist()
        data.append(row)
    return pd.DataFrame(data)


def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    args = get_args()
    
    # Batch size validation (ensure it's positive)
    if args.batch_size < 1:
        logging.warning(f"Invalid batch_size {args.batch_size}. Setting to 1.")
        args.batch_size = 1

    # Audio configuration (must match training)
    audio_conf = {
        'num_mel_bins': args.num_mel_bins,
        'mean': args.mean,
        'std': args.std,
        'target_length': args.target_length,
        'mode': 'unsupervised_train',
        'total_frame': args.total_frame,
        'im_res': args.im_res,
        'augmentation': False,
        'label_smooth': 0.0,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'custom',
        'skip_norm': False,
        'noise': False
    }
    
    # Create dataset
    if args.sharded_dataset_dir:
        logging.info(f"Using sharded dataset from {args.sharded_dataset_dir}")
        dataset = ShardedAudiosetDataset(
            shard_dir=args.sharded_dataset_dir,
            audio_conf=audio_conf,
            shuffle_shards=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=unsupervised_collate_fn
        )
    elif args.dataset_json:
        logging.info(f"Using legacy dataset from {args.dataset_json}")
        dataset = AudiosetDataset(
            dataset_json_file=args.dataset_json,
            audio_conf=audio_conf,
            label_csv=None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=unsupervised_collate_fn,
            pin_memory=True
        )
    else:
        raise ValueError("Must provide either --dataset_json or --sharded_dataset_dir")
    
    # Load model
    logging.info(f"Loading checkpoint from {args.checkpoint_path}...")
    model = CAVMAEModule.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    # Load optional labels
    labels_df = load_labels(args.label_csv)
    
    # Create evaluator
    logging.info(f"Masking enabled: {args.use_masking}")
    evaluator = DeepFakeEvaluator(
        model, 
        use_masking=args.use_masking,
        total_frames=args.total_frame,
        labels_df=labels_df
    )
    
    # Setup TensorBoard logger
    os.makedirs(args.output_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='tensorboard_eval',
        version=''
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,  # Single GPU for evaluation
        enable_checkpointing=False,
        logger=tb_logger,
        log_every_n_steps=10
    )
    
    # Run evaluation
    logging.info(f"Starting evaluation...")
    trainer.test(evaluator, dataloader)
    
    # Save results
    df = results_to_dataframe(evaluator.test_outputs)
    df.to_csv(args.output_csv, index=False)
    logging.info(f"Results saved to {args.output_csv}")
    
    # Print summary statistics
    logging.info("\n" + "="*60)
    logging.info("FINAL SUMMARY")
    logging.info("="*60)
    logging.info(f"Total videos: {len(df)}")
    logging.info(f"Sync Mean   : {df['sync_mean'].mean():.4f} ± {df['sync_mean'].std():.4f}")
    logging.info(f"Sync Min    : {df['sync_min'].mean():.4f} ± {df['sync_min'].std():.4f}")
    logging.info(f"Sync Std    : {df['sync_std'].mean():.4f} ± {df['sync_std'].std():.4f}")


if __name__ == "__main__":
    main()
