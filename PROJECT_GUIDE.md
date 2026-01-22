# CAV-MAE Sync for Deepfake Detection - Project Guide

> Research project adapting CAV-MAE Sync for unsupervised deepfake detection via audio-visual synchronization analysis.

---

## Table of Contents
1. [Research Goal](#research-goal)
2. [Architecture Overview](#architecture-overview)
3. [Key Modifications from Original](#key-modifications-from-original)
4. [Data Pipeline](#data-pipeline)
5. [Training Configuration](#training-configuration)
6. [Evaluation Approach](#evaluation-approach)
7. [Key Files Reference](#key-files-reference)
8. [Known Issues & Debugging](#known-issues--debugging)
9. [Quick Commands](#quick-commands)

---

## Research Goal

**Objective**: Detect deepfakes in an unsupervised manner by learning audio-visual alignment patterns.

**Hypothesis**: Deepfake videos exhibit subtle audio-visual desynchronization. By training a model to learn fine-grained alignment between audio and visual streams on real videos (VoxCeleb2), we can detect anomalies (low sync scores) that indicate potential fakes.

**Key Innovation**: Unlike the original CAV-MAE Sync which focuses on inter-instance alignment (matching audio/video across different videos in a batch), this project adds **intra-instance alignment** - sampling negative pairs from within the same video. This should improve the model's sensitivity to temporal misalignment within a single video.

---

## Architecture Overview

### Model: CAVMAE (Contrastive Audio-Visual Masked Autoencoder)

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│   Video: 10 seconds → 16 frames (224×224)                       │
│   Audio: 10 seconds → 16 mel-spectrogram segments               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PATCH EMBEDDING                               │
│   Audio: [B×16, 48, 128] → patches (48/16 × 128/16 = 24 patches)│
│   Video: [B×16, 3, 224, 224] → patches (14×14 = 196 patches)    │
│   + Positional Embeddings + Modality Embeddings                  │
│   + CLS Token (1 per modality) + Register Tokens (8 per mod.)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              MODALITY-SPECIFIC ENCODERS (11 layers each)         │
│   blocks_a: Audio transformer blocks                             │
│   blocks_v: Visual transformer blocks                            │
│   (Each block has modality-specific LayerNorm: norm1_a/v, etc.) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED ENCODER (1 layer)                     │
│   blocks_u: Shared transformer block                             │
│   Processes concatenated audio+visual tokens (for MAE)          │
│   Also processes each modality separately (for contrastive)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUTS                                  │
│                                                                  │
│   MAE Branch:          Contrastive Branch:                      │
│   ├─ Decoder (8 layers) ├─ cls_a: Audio CLS token [B×16, 768]   │
│   ├─ Reconstruct patches├─ cls_v: Visual CLS token [B×16, 768]  │
│   └─ MAE Loss           └─ Contrastive Loss (intra + inter)     │
└─────────────────────────────────────────────────────────────────┘
```

### CLS Tokens (Global Embeddings)

The "CLS tokens" in this codebase are **global representation tokens** (not classifier tokens):
- `cls_token_a`: Learnable audio embedding that attends to all audio patches
- `cls_token_v`: Learnable visual embedding that attends to all visual patches
- After encoding, `cls_a` and `cls_v` contain global representations of the audio and visual content
- **Sync Score**: Cosine similarity between `cls_a` and `cls_v` indicates alignment quality

### Register Tokens

- 8 register tokens per modality (total 16)
- Act as "memory" for the transformer to offload information
- Removed before final output (not used in loss computation)

---

## Key Modifications from Original

### 1. Intra-Instance Negative Sampling

**Original CAV-MAE Sync**: Only inter-video contrastive loss (match audio/video across batch)

**This Project**: Added intra-video contrastive loss
- For each video with 16 frames, the model must match each audio segment to its corresponding visual frame
- Negative samples include other frames from the **same video** (not just other videos)
- Controlled by `contrast_intra_weight` (0.7) and `contrast_inter_weight` (0.3)

Implementation in `src/models/cav_mae_sync.py:forward_contrastive()` lines 696-772:
```python
# mode="unsupervised_train" branch
# Intra-video: Match audio[i] to video[i] within same video (F×F matrix per video)
# Inter-video: Match audio[i] to video[i] across batch, masking same-video negatives
nce = contrast_intra_weight * loss_intra + contrast_inter_weight * loss_inter
```

### 2. Audio Segment Duration Change

**Original**: `audio_length=128` (longer audio context per frame)

**This Project**: `audio_length=48` (shorter, more localized audio)
- Each audio segment is `48 × 10ms = 480ms` centered on the frame timestamp
- This provides more precise temporal alignment
- Requires reinitialization of audio positional embeddings when loading pretrained weights

### 3. Bidirectional Contrastive Loss

Enabled via `--contrast_bidirect` flag:
- Audio→Video direction: Given audio, find matching video
- Video→Audio direction: Given video, find matching audio
- Final loss is average of both directions

---

## Data Pipeline

### Dataset: VoxCeleb2

- ~2 million real videos of celebrities speaking
- Used for unsupervised pretraining (no deepfake labels)
- Preprocessed into ~200 sharded `.pt` files

### Shard Format

Each shard contains ~1000 videos as a list of dictionaries:
```python
{
    'video_id': str,                    # Unique identifier
    'fbank': Tensor[max_len, 128],      # Full mel-spectrogram (float16, padded)
    'fbank_length': int,                # Actual length before padding
    'images': List[bytes],              # 16 JPEG-encoded frames
    'frame_indices': List[int],         # Frame numbers in original video
    'frame_timestamps_ms': List[int],   # Timestamps in milliseconds
}
```

### Dataloader Flow

```
ShardedAudiosetDataset (IterableDataset)
    │
    ├─ DDP: Partition shards across GPUs (rank-aware)
    ├─ Workers: Further partition shards within each GPU
    │
    └─ For each sample:
        ├─ Load full mel-spectrogram
        ├─ For each of 16 frames:
        │   ├─ Slice audio segment centered at frame timestamp
        │   ├─ Normalize audio: (fbank - mean) / std
        │   └─ Decode JPEG, apply transforms, normalize image
        └─ Yield: (fbanks[16, 48, 128], images[16, 3, 224, 224], video_id, frame_indices)
```

### Audio Slicing Logic

`slice_fbank_at_timestamp()` in `src/dataloader_sharded.py:101-143`:
- Frame shift: 10ms per mel-spectrogram frame
- For timestamp T ms: center_frame = T / 10
- Extract `[center - 24 : center + 24]` = 48 frames = 480ms window
- Edge padding if segment extends beyond audio boundaries

---

## Training Configuration

### Current Settings (train_unsupervised_sharded.sbatch)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 18 | Videos per GPU per step |
| `total_frame` | 16 | Frames sampled per video |
| `audio_length` | 48 | Mel-spec frames per audio segment |
| `target_length` | 48 | Must equal audio_length |
| `epochs` | 10 | Training epochs |
| `lr` | 1e-4 | Learning rate |
| `warmup_epochs` | 1 | Linear warmup |
| `gradient_accumulation_steps` | 8 | Effective batch = 18×8×2 = 288 videos |
| `mask_ratio_a` | 0.75 | Audio masking for MAE |
| `mask_ratio_v` | 0.75 | Video masking for MAE |
| `mae_loss_weight` | 1.0 | MAE reconstruction loss weight |
| `contrast_loss_weight` | 0.5 | Contrastive loss weight |
| `contrast_intra_weight` | 0.7 | Intra-video contrastive weight |
| `contrast_inter_weight` | 0.3 | Inter-video contrastive weight |
| `contrast_bidirect` | True | Bidirectional contrastive loss |
| `cls_token` | True | Use CLS tokens |
| `num_register_tokens` | 8 | Register tokens per modality |

### Hardware (SLURM)

- 2× NVIDIA A40 GPUs
- 5 CPUs per GPU
- 64GB RAM
- 72 hour time limit

### Effective Batch Calculation

```
Per step per GPU: 18 videos × 16 frames = 288 samples
With accumulation: 288 × 8 = 2304 samples per GPU
With 2 GPUs: 2304 × 2 = 4608 samples effective batch
```

---

## Evaluation Approach

### Dataset: FakeAVCeleb

Used for evaluating deepfake detection performance (not for training).

### Sync Score Computation

For each video:
1. Sample 16 frames with corresponding audio segments
2. Extract `cls_a` and `cls_v` for each frame
3. Compute cosine similarity: `sim[i] = cos(cls_a[i], cls_v[i])`
4. Aggregate across frames using multiple strategies:

| Aggregation | Formula | Intuition |
|-------------|---------|-----------|
| `mean` | `mean(sim)` | Average alignment quality |
| `min` | `min(sim)` | Worst-case alignment |
| `p10` | `percentile(sim, 10)` | Robust worst-case |
| `p25` | `percentile(sim, 25)` | Lower quartile |
| `variance` | `var(sim)` | Consistency of alignment |

### Expected Behavior

- **Real videos**: High sync scores (close to 1.0), low variance
- **Deepfakes**: Lower sync scores, higher variance (inconsistent alignment)

### Evaluation Metrics

- AUC-ROC
- EER (Equal Error Rate)
- Accuracy at various thresholds

---

## Key Files Reference

### Core Model & Training

| File | Purpose |
|------|---------|
| `src/models/cav_mae_sync.py` | CAVMAE model (1740 lines) |
| `train_unsupervised.py` | PyTorch Lightning training script |
| `src/fakesync_config.py` | Configuration dataclass with validation |
| `src/dataloader_sharded.py` | Sharded IterableDataset |
| `src/dataloader_sync.py` | Legacy JSON-based dataset |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/train_unsupervised_sharded.sbatch` | Multi-GPU SLURM training |
| `scripts/train_unsupervised.sbatch` | Single-GPU training |
| `eval_deepfake.py` | Deepfake evaluation script |

### Preprocessing

| File | Purpose |
|------|---------|
| `preprocess/create_sharded_dataset.py` | Create .pt shards from videos |
| `preprocess/create_json_vox.py` | Generate VoxCeleb2 JSON |
| `get_norm_stats.py` | Compute dataset normalization stats |

### Tests

| File | Purpose |
|------|---------|
| `tests/test_sync.py` | Validate audio-visual sync in shards |
| `tests/test_normalization.py` | Test audio normalization |
| `tests/verify_sharded_dataloader_parity.py` | Test dataloader correctness |

---

## Known Issues & Debugging

### FIXED: Multi-GPU Training Hangs After 30-45 Minutes

**Root Cause**: With `IterableDataset` and DDP, different ranks had different numbers of samples due to unequal shard sizes. When one rank exhausted its data, it moved to epoch-end sync operations while the other rank was still doing gradient sync → NCCL ALLREDUCE mismatch → deadlock → timeout.

**The Fix** (implemented in `CAVMAEDataModule`):

1. Dataset creation moved to `LightningDataModule.setup()` which runs AFTER DDP initialization
2. Each rank counts actual samples in its assigned shards
3. Ranks synchronize via `dist.all_reduce(MIN)` to find the minimum sample count
4. All ranks limit iteration to this minimum, ensuring equal batch counts

**Key Code Changes**:
- `src/dataloader_sharded.py`: Added `rank`, `world_size`, `max_samples` parameters
- `train_unsupervised.py`: Added `CAVMAEDataModule` class that handles DDP sync

**Verification**: You should see these logs during training:
```
DataModule setup: rank=0, world_size=2
Rank 0: counted 395000 samples in assigned shards
Rank 0: synced epoch length to 390000 samples (was 395000)
```

### Audio Length Mismatch Warning

When loading pretrained weights with different `audio_length`:
- Positional embeddings have shape mismatch
- Training script auto-reinitializes with sin-cos encoding
- This is expected behavior, not an error

### DDP find_unused_parameters

The model uses `find_unused_parameters=True` because:
- Contrastive heads may be conditionally unused
- Different code paths in `forward()` based on `cls_token` flag

---

## Quick Commands

### Start Training (Multi-GPU)
```bash
sbatch scripts/train_unsupervised_sharded.sbatch
```

### Start Training (Single GPU - for debugging)
```bash
# Modify sbatch to use 1 GPU, or run directly:
python train_unsupervised.py \
    --sharded_dataset_dir /path/to/shards \
    --batch_size 12 \
    --total_frame 16 \
    --audio_length 48 \
    --target_length 48 \
    --epochs 1 \
    --cls_token \
    --fast_dev_run  # Quick test
```

### Monitor Training
```bash
# TensorBoard
tensorboard --logdir outputs/checkpoints/*/tensorboard

# SLURM logs
tail -f /storage/slurm/hunecke/fakesync/slurm_logs/slurm-cav-mae-sync-train-sharded-*.out
```

### Test Data Sync
```bash
python tests/test_sync.py --shard_path /path/to/shard_0000.pt
```

### Compute Normalization Stats (for new dataset)
```bash
python get_norm_stats.py --shard_dir /path/to/shards
```

---

## Critical Constraints

1. **audio_length % 16 == 0**: Patch size divides evenly
   - Valid: 48, 64, 80, 96, 112, 128
   - Current: 48

2. **target_length == audio_length**: Must match for model

3. **embed_dim % 12 == 0**: Divisible by num_heads
   - Default: 768 (768/12 = 64)

4. **Normalization stats**: Current values are for VoxCeleb2
   - mean: -6.166528
   - std: 3.483568
   - Recompute for different datasets

---

## Project Status

- [x] Adapted CAV-MAE Sync architecture
- [x] Implemented intra-instance negative sampling
- [x] Changed audio segment duration (128 → 48)
- [x] Set up sharded data pipeline
- [x] Multi-GPU training setup
- [x] **Fixed multi-GPU DDP deadlock** (via `CAVMAEDataModule` with epoch sync)
- [ ] Complete training on VoxCeleb2
- [ ] Evaluate on FakeAVCeleb
- [ ] Report metrics (AUC, EER, etc.)

---

*Last updated: January 2026*
