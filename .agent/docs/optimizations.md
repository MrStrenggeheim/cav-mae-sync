# Training Optimizations

Applied optimizations for CAV-MAE training on A40 cluster.

## Flash Attention

**What**: Uses PyTorch 2.0+ `scaled_dot_product_attention` with Flash kernel.

**Where**: `train_unsupervised.py` lines 171-174

**Potential Issues**:
- If training crashes with CUDA errors related to attention, disable with:
  ```python
  torch.backends.cuda.enable_flash_sdp(False)
  ```
- Flash Attention requires head_dim to be a power of 2 and <= 128. CAV-MAE uses 768/12=64, which is fine.
- May behave differently with unusual sequence lengths.

---

## torch.compile

**What**: JIT-compiles the model for faster execution via TorchDynamo.

**Where**: `train_unsupervised.py` after model creation

**Potential Issues**:
- First iteration is slow (compilation overhead). This is normal.
- If you see `torch._dynamo` errors or recompilation warnings:
  1. The model has dynamic shapes (unlikely here)
  2. Workaround: remove `torch.compile()` call temporarily
- Checkpointing works, but resuming from checkpoint may re-trigger compilation.
- If using `fast_dev_run`, the compile overhead will dominate the short run time.

---

## bfloat16 Precision

**What**: Uses bf16-mixed precision on Ampere+ GPUs (compute >= 8.0).

**Where**: `train_unsupervised.py` lines 269-279

**Potential Issues**:
- If loss becomes NaN or training is unstable on older code:
  - bf16 has same exponent range as fp32 (less under/overflow risk than fp16)
  - Still check for exploding gradients
- If GPU doesn't support bf16 (pre-Ampere), code falls back to fp16 automatically.
- A40 is Ampere (compute 8.6), so bf16 will be used.

---

## Prefetch Factor

**What**: Increases DataLoader prefetching from default 2 to 4 per worker.

**Where**: `train_unsupervised.py` DataLoader creation

**Potential Issues**:
- Higher memory usage (4 batches queued per worker instead of 2)
- If OOM during data loading, reduce to default:
  ```python
  prefetch_factor=2
  ```

---

## mmap Shard Loading

**What**: Uses memory-mapped file access instead of full RAM loading.

**Where**: `dataloader_sharded.py` line 205

**Potential Issues**:
- mmap relies on OS page cache; may be slower if cache is cold
- On NFS/network storage, mmap behavior depends on the filesystem
- If shard loading becomes slow or produces IO errors:
  1. Check if network storage supports mmap efficiently
  2. Fall back to non-mmap: remove `mmap=True`
- Tested locally: works with JPEG bytes in shard format

---

## Summary of Fallback Options

| Issue | Check | Fix |
|-------|-------|-----|
| CUDA attention errors | Logs mention "flash" or "sdp" | Disable Flash Attention |
| torch.compile errors | `torch._dynamo` tracebacks | Remove `torch.compile()` |
| bf16 NaN/instability | Check `device_capability` print | Training script has auto-fallback |
| OOM during loading | Error during prefetch | Reduce `prefetch_factor` |
| Slow shard IO | Training stalls at load | Remove `mmap=True` |
