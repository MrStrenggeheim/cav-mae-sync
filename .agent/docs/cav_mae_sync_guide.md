# CAV-MAE Sync Project Guide

> Deepfake detection via audio-visual synchronization analysis using CAV-MAE architecture.

## Architecture

**Model**: CAV-MAE (Contrastive Audio-Visual Masked Autoencoder)
- Dual-stream encoder (audio + visual) with shared transformer blocks
- MAE reconstruction loss + contrastive loss (intra/inter-video)
- Uses CLS tokens and register tokens for improved representations

**Key Insight**: Deepfakes often have subtle audio-visual desync. The model learns synchronization patterns, then anomalies indicate potential fakes.

---

## Key Files

| File | Purpose |
|------|---------|
| `train_unsupervised.py` | Main training script (PyTorch Lightning) |
| `eval_deepfake.py` | Evaluation for deepfake detection |
| `src/dataloader_sharded.py` | IterableDataset for sharded data |
| `src/models/cav_mae_sync.py` | CAVMAE model (~1600 lines) |
| `preprocess/create_sharded_dataset.py` | Preprocessing pipeline |

---

## Training Configuration

### SLURM Defaults (A40 GPU)
```bash
--batch_size 12              # Videos per batch (×16 frames = 192 samples)
--total_frame 16             # Frames sampled per video
--target_length 48           # Fbank frames per audio slice (MUST = audio_length)
--audio_length 48            # Model's audio patch dimension (divisible by 16!)
--log_freq 100               # TensorBoard log interval
--contrast_loss_weight 0.1   # Contrastive vs MAE loss ratio
--cls_token                  # Use CLS tokens
--num_register_tokens 8      # Register tokens per modality
```

### Multi-GPU (NEW)
```bash
#SBATCH --gres=gpu:a40:2     # Request 2 GPUs
#SBATCH --cpus-per-task=10   # 5 CPUs per GPU
```
- Dataset automatically partitions shards across GPUs then workers
- DDP with `find_unused_parameters=True` (contrastive heads may be unused)

---

## Critical Constraints

### Audio Length Divisibility
`audio_length` and `target_length` **MUST be divisible by 16** (patch size).
```
Valid: 48, 64, 80, 96, 112, 128, ...
Invalid: 42, 50, 100, ...
```

### Video-Audio Synchronization
Frame timestamps must be ≤ `fbank_length × 10ms`. See `.agent/workflows/sync_requirements.md`.

**Always run after preprocessing changes:**
```bash
python tests/test_sync.py --shard_path <shard.pt>
```

---

## Known Issues / Quirks

1. **Contrastive heads**: Model has conditional `contrastive_heads` blocks. If `contrastive_heads=False` in config, those parameters are unused → DDP error without `find_unused_parameters=True`.

2. **Normalization stats**: Default `mean=-4.05, std=4.07` are for VoxCeleb2. Recompute for new datasets via `get_norm_stats.py`.

3. **Legacy code**: `src/dataloader.py`, `src/avceleb_dataloader.py`, and various eval scripts are legacy. Focus on `*_sharded` variants.

4. **Shard loading**: Uses `torch.load(..., weights_only=False)` for flexibility. Shards contain dicts with JPEG bytes.

---

## File I/O Optimization

**Problem**: Network storage on cluster causes slow file access.

**Solution**: Sharded dataset minimizes file operations:
- ~1000 videos per shard (one `.pt` file)
- Shard loaded once, samples yielded in memory
- Workers partition shards (no overlap)

---

## Testing Checklist

Before submitting to cluster:
1. `python tests/test_sync.py --shard_path <path>` - Sync validation
2. `python -c "from src.dataloader_sharded import *; print('OK')"` - Import check
3. Verify `--audio_length % 16 == 0` in batch script
