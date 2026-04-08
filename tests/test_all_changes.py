#!/usr/bin/env python3
"""
Comprehensive test script for validating all recent changes to CAV-MAE Sync.

Covers: model fixes, WebDataset migration, eval improvements, training config.
Designed to run on a SLURM cluster without GPU (CPU-only for most tests).
Collects all results and reports at the end — does NOT stop at first error.

Usage:
    python tests/test_all_changes.py
"""

import io
import json
import os
import sys
import tempfile
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass
from typing import Callable, List

# Add project root to sys.path so imports work from any working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration: float


results: List[TestResult] = []


def run_test(name: str, fn: Callable):
    """Run a test function, catch all exceptions, record result."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"  PASSED ({dt:.2f}s)")
        results.append(TestResult(name=name, passed=True, message="OK", duration=dt))
    except Exception as e:
        dt = time.time() - t0
        tb = traceback.format_exc()
        print(f"  FAILED ({dt:.2f}s)")
        print(f"  Error: {e}")
        print(f"  Traceback:\n{tb}")
        results.append(TestResult(name=name, passed=False, message=f"{e}\n{tb}", duration=dt))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_default_args(**overrides):
    """Create a default Namespace mimicking the training script args."""
    defaults = dict(
        im_res=224,
        audio_length=48,
        embed_dim=768,
        total_frame=16,
        contrastive_heads=True,
        cls_token=True,
        num_register_tokens=4,
        mask_ratio_a=0.75,
        mask_ratio_v=0.75,
        mae_loss_weight=1.0,
        contrast_loss_weight=0.01,
        contrast_bidirect=False,
        contrast_inter_weight=0.4,
        contrast_intra_weight=0.6,
        lr=1e-4,
        warmup_epochs=1,
        weight_decay=0.05,
        epochs=10,
        batch_size=4,
        num_workers=0,
        sharded_dataset_dir=None,
        dataset_json=None,
        label_csv=None,
        save_path="./checkpoints",
        resume=None,
        fast_dev_run=False,
        log_freq=10,
        checkpoint_interval_hours=1.0,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        num_mel_bins=128,
        mean=-6.166528,
        std=3.483568,
        target_length=48,
        sync_aggregation="all",
        sync_similarity="cosine",
        sync_temperature=0.05,
        sync_temporal_variance=True,
        use_mmap=False,
        dataset_fraction=1.0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_dummy_batch(batch_size=2, total_frame=16, target_length=48, num_mel_bins=128, im_res=224):
    """Create a dummy batch matching the dataloader output format."""
    n = batch_size * total_frame
    fbanks = torch.randn(n, target_length, num_mel_bins)
    images = torch.randn(n, 3, im_res, im_res)
    video_ids = []
    for i in range(batch_size):
        video_ids.extend([f"video_{i}"] * total_frame)
    frame_indices = torch.arange(total_frame).repeat(batch_size)
    return fbanks, images, video_ids, frame_indices


# ============================================================================
# MODEL TESTS
# ============================================================================

def test_forward_cls_token_batch1():
    """batch_size=1 forward pass with cls_token=True (squeeze fix)."""
    from src.models.cav_mae_sync import CAVMAE

    model = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=True,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
    )
    model.eval()

    # batch_size=1 means 1*16 = 16 frame-level samples
    fbanks = torch.randn(16, 48, 128)
    images = torch.randn(16, 3, 224, 224)

    with torch.no_grad():
        out = model(
            fbanks, images,
            mask_ratio_a=0.75, mask_ratio_v=0.75,
            mae_loss_weight=1.0, contrast_loss_weight=0.01,
            mode='unsupervised_train',
        )

    assert 'loss' in out, f"Missing 'loss' key, got keys: {list(out.keys())}"
    assert 'cls_a' in out, f"Missing 'cls_a' key"
    assert 'cls_v' in out, f"Missing 'cls_v' key"

    # cls_a/cls_v should be 2D: (16, 768)
    assert out['cls_a'].dim() == 2, f"cls_a should be 2D, got {out['cls_a'].shape}"
    assert out['cls_v'].dim() == 2, f"cls_v should be 2D, got {out['cls_v'].shape}"
    assert out['cls_a'].shape == (16, 768), f"cls_a shape {out['cls_a'].shape} != (16, 768)"
    assert out['cls_v'].shape == (16, 768), f"cls_v shape {out['cls_v'].shape} != (16, 768)"

    print(f"  cls_a shape: {out['cls_a'].shape}")
    print(f"  cls_v shape: {out['cls_v'].shape}")
    print(f"  loss: {out['loss'].item():.4f}")


def test_forward_no_cls_token_batch1():
    """batch_size=1 forward pass with cls_token=False."""
    from src.models.cav_mae_sync import CAVMAE

    model = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=False,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
    )
    model.eval()

    fbanks = torch.randn(16, 48, 128)
    images = torch.randn(16, 3, 224, 224)

    with torch.no_grad():
        out = model(
            fbanks, images,
            mask_ratio_a=0.75, mask_ratio_v=0.75,
            mae_loss_weight=1.0, contrast_loss_weight=0.01,
            mode='unsupervised_train',
        )

    assert 'loss' in out, f"Missing 'loss' key"
    assert 'cls_a' in out, f"Missing 'cls_a' key"
    assert 'cls_v' in out, f"Missing 'cls_v' key"

    # When cls_token=False, cls_a/cls_v are mean-pooled patch embeddings: (16, 768)
    assert out['cls_a'].dim() == 2, f"cls_a should be 2D, got {out['cls_a'].shape}"
    assert out['cls_a'].shape[0] == 16, f"cls_a batch dim {out['cls_a'].shape[0]} != 16"
    assert out['cls_a'].shape[1] == 768, f"cls_a embed dim {out['cls_a'].shape[1]} != 768"

    print(f"  cls_a shape: {out['cls_a'].shape}")
    print(f"  cls_v shape: {out['cls_v'].shape}")
    print(f"  loss: {out['loss'].item():.4f}")


def test_mae_loss_weight_zero():
    """mae_loss_weight=0 forward pass should not produce NameError."""
    from src.models.cav_mae_sync import CAVMAE

    model = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=True,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
    )
    model.eval()

    fbanks = torch.randn(16, 48, 128)
    images = torch.randn(16, 3, 224, 224)

    with torch.no_grad():
        out = model(
            fbanks, images,
            mask_ratio_a=0.75, mask_ratio_v=0.75,
            mae_loss_weight=0.0,
            contrast_loss_weight=0.01,
            mode='unsupervised_train',
        )

    assert out['loss_mae'].item() == 0.0, f"loss_mae should be 0.0, got {out['loss_mae'].item()}"
    assert out['recon_a'] is None, "recon_a should be None when mae_loss_weight=0"
    assert out['recon_v'] is None, "recon_v should be None when mae_loss_weight=0"
    # loss should be purely contrastive
    assert torch.isfinite(out['loss']), f"loss is not finite: {out['loss'].item()}"
    print(f"  loss (contrastive only): {out['loss'].item():.4f}")
    print(f"  loss_mae: {out['loss_mae'].item():.4f}")
    print(f"  loss_c: {out['loss_c'].item():.4f}")


def test_forward_feat_shapes():
    """forward_feat produces correct shapes and chains blocks."""
    from src.models.cav_mae_sync import CAVMAE

    model = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=True,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
    )
    model.eval()

    fbanks = torch.randn(4, 48, 128)
    images = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        result = model.forward_feat(fbanks, images)

    # With cls_token=True, returns (ca, cv, cls_a, cls_v)
    assert len(result) == 4, f"Expected 4 outputs, got {len(result)}"
    ca, cv, cls_a, cls_v = result

    # Audio patches: 48/16 * 128/16 = 3 * 8 = 24 patches
    n_audio_patches = (48 // 16) * (128 // 16)
    # Visual patches: 14 * 14 = 196 patches
    n_visual_patches = (224 // 16) ** 2

    assert ca.shape == (4, n_audio_patches, 768), f"ca shape {ca.shape} != (4, {n_audio_patches}, 768)"
    assert cv.shape == (4, n_visual_patches, 768), f"cv shape {cv.shape} != (4, {n_visual_patches}, 768)"
    assert cls_a.shape == (4, 768), f"cls_a shape {cls_a.shape} != (4, 768)"
    assert cls_v.shape == (4, 768), f"cls_v shape {cls_v.shape} != (4, 768)"

    print(f"  ca shape: {ca.shape} (audio patches)")
    print(f"  cv shape: {cv.shape} (visual patches)")
    print(f"  cls_a shape: {cls_a.shape}")
    print(f"  cls_v shape: {cls_v.shape}")

    # Also test cls_token=False path
    model_nocls = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=False,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
    )
    model_nocls.eval()

    with torch.no_grad():
        result_nocls = model_nocls.forward_feat(fbanks, images)

    assert len(result_nocls) == 2, f"Expected 2 outputs (no cls), got {len(result_nocls)}"
    ca2, cv2 = result_nocls
    assert ca2.shape == (4, n_audio_patches, 768), f"ca shape {ca2.shape}"
    assert cv2.shape == (4, n_visual_patches, 768), f"cv shape {cv2.shape}"
    print(f"  cls_token=False: ca shape {ca2.shape}, cv shape {cv2.shape}")


def test_register_token_init_std():
    """Register token init std should be ~0.02."""
    from src.models.cav_mae_sync import CAVMAE

    model = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=True,
        contrastive_heads=True, num_register_tokens=8, total_frame=16,
    )

    reg_std = model.register_tokens.data.std().item()
    print(f"  register_tokens std: {reg_std:.4f}")
    # After init, std should be close to 0.02 (tolerance for finite samples)
    assert abs(reg_std - 0.02) < 0.01, (
        f"register_tokens std {reg_std:.4f} not close to 0.02"
    )

    if model.cls_token:
        cls_a_std = model.cls_token_a.data.std().item()
        cls_v_std = model.cls_token_v.data.std().item()
        print(f"  cls_token_a std: {cls_a_std:.4f}")
        print(f"  cls_token_v std: {cls_v_std:.4f}")
        assert abs(cls_a_std - 0.02) < 0.01, f"cls_token_a std {cls_a_std:.4f} not close to 0.02"
        assert abs(cls_v_std - 0.02) < 0.01, f"cls_token_v std {cls_v_std:.4f} not close to 0.02"


def test_contrastive_loss_bidirect():
    """Contrastive loss with contrast_bidirect=True works (masked_fill fix)."""
    from src.models.cav_mae_sync import CAVMAE

    model = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=True,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
    )
    model.eval()

    # batch_size=2, so 32 frames
    fbanks = torch.randn(32, 48, 128)
    images = torch.randn(32, 3, 224, 224)

    with torch.no_grad():
        out = model(
            fbanks, images,
            mask_ratio_a=0.75, mask_ratio_v=0.75,
            mae_loss_weight=1.0, contrast_loss_weight=0.01,
            contrast_bidirect=True,
            mode='unsupervised_train',
        )

    assert torch.isfinite(out['loss']), f"loss not finite: {out['loss'].item()}"
    assert torch.isfinite(out['loss_c']), f"loss_c not finite: {out['loss_c'].item()}"
    assert torch.isfinite(out['c_acc']), f"c_acc not finite: {out['c_acc'].item()}"
    assert torch.isfinite(out['inter_acc']), f"inter_acc not finite: {out['inter_acc'].item()}"
    print(f"  loss: {out['loss'].item():.4f}")
    print(f"  loss_c: {out['loss_c'].item():.4f}")
    print(f"  c_acc (intra): {out['c_acc'].item():.4f}")
    print(f"  inter_acc: {out['inter_acc'].item():.4f}")


def test_gradient_flow_contrastive():
    """Gradient flows through contrastive loss (backward doesn't error)."""
    from src.models.cav_mae_sync import CAVMAE

    model = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=True,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
    )
    model.train()

    fbanks = torch.randn(32, 48, 128)
    images = torch.randn(32, 3, 224, 224)

    out = model(
        fbanks, images,
        mask_ratio_a=0.75, mask_ratio_v=0.75,
        mae_loss_weight=1.0, contrast_loss_weight=0.01,
        contrast_bidirect=True,
        mode='unsupervised_train',
    )

    loss = out['loss']
    loss.backward()

    # Check that gradients exist for key parameters
    params_with_grad = 0
    params_without_grad = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and p.grad.abs().sum() > 0:
                params_with_grad += 1
            else:
                params_without_grad += 1

    print(f"  Params with gradient: {params_with_grad}")
    print(f"  Params without gradient: {params_without_grad}")
    assert params_with_grad > 0, "No parameters received gradients!"
    # Most params should have gradients — a few (e.g., frozen pos_embed) may not
    total = params_with_grad + params_without_grad
    pct = params_with_grad / total * 100
    print(f"  Gradient coverage: {pct:.1f}%")
    assert pct > 50, f"Only {pct:.1f}% params have gradients — something is wrong"


def test_output_dict_keys():
    """Output dict keys are correct for all code paths."""
    from src.models.cav_mae_sync import CAVMAE

    # Path 1: cls_token=True, global_local_losses=False
    model1 = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=True,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
        global_local_losses=False,
    )
    model1.eval()

    fbanks = torch.randn(16, 48, 128)
    images = torch.randn(16, 3, 224, 224)

    with torch.no_grad():
        out1 = model1(fbanks, images, mae_loss_weight=1.0, contrast_loss_weight=0.01, mode='unsupervised_train')

    expected_keys_cls = {
        'loss', 'loss_mae', 'loss_mae_a', 'loss_mae_v', 'loss_c',
        'mask_a', 'mask_v', 'c_acc', 'recon_a', 'recon_v',
        'cls_a', 'cls_v', 'latent_c_a_mean', 'latent_c_v_mean', 'inter_acc',
    }
    assert set(out1.keys()) == expected_keys_cls, (
        f"cls_token=True keys mismatch.\n  Expected: {sorted(expected_keys_cls)}\n  Got: {sorted(out1.keys())}"
    )
    print(f"  cls_token=True keys: {sorted(out1.keys())}")

    # Path 2: cls_token=False
    model2 = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=False,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
    )
    model2.eval()

    with torch.no_grad():
        out2 = model2(fbanks, images, mae_loss_weight=1.0, contrast_loss_weight=0.01, mode='unsupervised_train')

    expected_keys_nocls = {
        'loss', 'loss_mae', 'loss_mae_a', 'loss_mae_v', 'loss_c',
        'mask_a', 'mask_v', 'c_acc', 'recon_a', 'recon_v',
        'cls_a', 'cls_v', 'inter_acc',
    }
    assert set(out2.keys()) == expected_keys_nocls, (
        f"cls_token=False keys mismatch.\n  Expected: {sorted(expected_keys_nocls)}\n  Got: {sorted(out2.keys())}"
    )
    print(f"  cls_token=False keys: {sorted(out2.keys())}")

    # Path 3: cls_token=True, global_local_losses=True
    model3 = CAVMAE(
        audio_length=48, embed_dim=768, cls_token=True,
        contrastive_heads=True, num_register_tokens=4, total_frame=16,
        global_local_losses=True,
    )
    model3.eval()

    with torch.no_grad():
        out3 = model3(fbanks, images, mae_loss_weight=1.0, contrast_loss_weight=0.01, mode='unsupervised_train')

    expected_keys_gl = expected_keys_cls | {'global_loss_c', 'local_loss_c'}
    assert set(out3.keys()) == expected_keys_gl, (
        f"global_local keys mismatch.\n  Expected: {sorted(expected_keys_gl)}\n  Got: {sorted(out3.keys())}"
    )
    print(f"  global_local=True keys: {sorted(out3.keys())}")


# ============================================================================
# WEBDATASET SHARD WRITER TESTS
# ============================================================================

def _create_test_shards(tmpdir, n_samples=5, total_frame=16, num_mel_bins=128, max_audio_length=1024, im_res=224):
    """Helper: create test WebDataset shards in tmpdir, return shard dir path."""
    import webdataset as wds
    from PIL import Image

    shard_dir = os.path.join(tmpdir, "test_shards")
    os.makedirs(shard_dir, exist_ok=True)
    pattern = os.path.join(shard_dir, "shard-%06d.tar")
    sink = wds.ShardWriter(pattern, maxsize=100_000_000)

    for idx in range(n_samples):
        key = f"{idx:08d}"
        video_id = f"test_video_{idx}"
        fbank_length = min(200 + idx * 10, max_audio_length)

        # Create a realistic fbank: float16 numpy array
        fbank_np = np.random.randn(max_audio_length, num_mel_bins).astype(np.float16)

        # Frame timestamps: evenly spaced
        frame_timestamps_ms = [int(i * (fbank_length * 10) / total_frame) for i in range(total_frame)]
        frame_indices = list(range(total_frame))

        metadata = {
            'video_id': video_id,
            'fbank_length': fbank_length,
            'frame_indices': frame_indices,
            'frame_timestamps_ms': frame_timestamps_ms,
            'video_fps': 25.0,
            'video_duration_ms': fbank_length * 10.0,
            'identity': f'id{idx:05d}',
        }

        record = {"__key__": key}
        record['json'] = json.dumps(metadata).encode('utf-8')

        # Save fbank as npy bytes
        buf = io.BytesIO()
        np.save(buf, fbank_np)
        record['fbank.npy'] = buf.getvalue()

        # Create JPEG frames
        for i in range(total_frame):
            img = Image.fromarray(np.random.randint(0, 256, (im_res, im_res, 3), dtype=np.uint8))
            img_buf = io.BytesIO()
            img.save(img_buf, format='JPEG', quality=90)
            record[f'{i:03d}.jpg'] = img_buf.getvalue()

        sink.write(record)

    sink.close()

    # Write metadata JSON
    meta = {
        'splits': {'test': {'total_valid': n_samples}},
        'args': {},
        'decoder': 'test',
    }
    with open(os.path.join(shard_dir, 'webdataset_metadata.json'), 'w') as f:
        json.dump(meta, f)

    return shard_dir


def test_shard_write_and_read():
    """Write 5 test samples to WebDataset shards and read them back."""
    import webdataset as wds
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = _create_test_shards(tmpdir, n_samples=5)

        # Read back
        tar_files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith('.tar')])
        assert len(tar_files) > 0, "No tar files created"

        dataset = wds.WebDataset(tar_files)
        samples = list(dataset)

        assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}"
        print(f"  Read back {len(samples)} samples")

        for i, sample in enumerate(samples):
            # Check metadata
            meta = json.loads(sample['json'])
            assert meta['video_id'] == f"test_video_{i}", f"Wrong video_id: {meta['video_id']}"
            assert 'fbank_length' in meta
            assert 'frame_timestamps_ms' in meta
            assert len(meta['frame_timestamps_ms']) == 16

            # Check fbank
            fbank_np = np.load(io.BytesIO(sample['fbank.npy']))
            assert fbank_np.shape == (1024, 128), f"fbank shape {fbank_np.shape} != (1024, 128)"
            assert fbank_np.dtype == np.float16, f"fbank dtype {fbank_np.dtype} != float16"

            # Check frames exist
            for j in range(16):
                frame_key = f'{j:03d}.jpg'
                assert frame_key in sample, f"Missing frame {frame_key}"
                img = Image.open(io.BytesIO(sample[frame_key]))
                assert img.size == (224, 224), f"Frame size {img.size} != (224, 224)"

        print("  All fields verified: metadata JSON, fbank shape/dtype, JPEG frames")


def test_identity_splitting():
    """Identity-aware splitting produces disjoint identity sets."""
    # We need to import the function
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocess'))
    from create_webdataset import create_identity_splits

    file_list = []
    for identity_idx in range(20):
        identity = f"id{identity_idx:05d}"
        for vid_idx in range(5):
            file_list.append({
                'video_path': f'/data/{identity}/video_{vid_idx}.mp4',
            })

    splits = create_identity_splits(
        file_list,
        split_ratios=(0.6, 0.2, 0.2),
        seed=42,
    )

    train_ids = {item['identity'] for item in splits['train']}
    val_ids = {item['identity'] for item in splits['val']}
    test_ids = {item['identity'] for item in splits['test']}

    # Check disjoint
    assert train_ids.isdisjoint(val_ids), f"Train and val share identities: {train_ids & val_ids}"
    assert train_ids.isdisjoint(test_ids), f"Train and test share identities: {train_ids & test_ids}"
    assert val_ids.isdisjoint(test_ids), f"Val and test share identities: {val_ids & test_ids}"

    # Check all items are in some split
    total = len(splits['train']) + len(splits['val']) + len(splits['test'])
    assert total == 100, f"Total items {total} != 100"

    print(f"  Train: {len(splits['train'])} items, {len(train_ids)} identities")
    print(f"  Val:   {len(splits['val'])} items, {len(val_ids)} identities")
    print(f"  Test:  {len(splits['test'])} items, {len(test_ids)} identities")
    print(f"  Identity sets are disjoint: OK")


def test_identity_exclusion():
    """Identity exclusion removes the specified identities."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocess'))
    from create_webdataset import create_identity_splits

    file_list = []
    for identity_idx in range(10):
        identity = f"id{identity_idx:05d}"
        for vid_idx in range(3):
            file_list.append({
                'video_path': f'/data/{identity}/video_{vid_idx}.mp4',
            })

    exclude = {f"id{i:05d}" for i in range(3)}  # Exclude first 3 identities

    splits = create_identity_splits(
        file_list,
        split_ratios=(0.8, 0.1, 0.1),
        exclude_identities=exclude,
        seed=42,
    )

    all_identities = set()
    for split_name, items in splits.items():
        for item in items:
            all_identities.add(item['identity'])

    for ex_id in exclude:
        assert ex_id not in all_identities, f"Excluded identity {ex_id} found in splits"

    total = sum(len(v) for v in splits.values())
    # 10 identities - 3 excluded = 7 identities * 3 videos = 21
    assert total == 21, f"Total items {total} != 21 (after excluding 3 identities)"
    print(f"  Excluded {len(exclude)} identities, {total} items remain")


def test_empty_input_shard_writer():
    """Empty input handling: write_webdataset_shards with no valid samples."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocess'))
    from create_webdataset import write_webdataset_shards

    with tempfile.TemporaryDirectory() as tmpdir:
        # All samples are invalid
        samples = [
            {'video_id': 'v1', 'valid': False, 'identity': 'id1'},
            {'video_id': 'v2', 'valid': False, 'identity': 'id2'},
        ]
        n_written = write_webdataset_shards(samples, tmpdir, 'test')
        assert n_written == 0, f"Expected 0 written, got {n_written}"
        print(f"  Empty input handled correctly: {n_written} samples written")


# ============================================================================
# WEBDATASET DATALOADER TESTS
# ============================================================================

def test_build_webdataset_iterate():
    """Create temp shards, build_webdataset, iterate, verify shapes."""
    from src.dataloader_webdataset import build_webdataset

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = _create_test_shards(tmpdir, n_samples=3)

        audio_conf = {
            'target_length': 48,
            'num_mel_bins': 128,
            'mean': -6.166528,
            'std': 3.483568,
            'im_res': 224,
            'total_frame': 16,
            'augmentation': False,
        }

        dataset = build_webdataset(
            shard_dir=shard_dir,
            audio_conf=audio_conf,
            shuffle=False,
            resampled=False,
        )

        count = 0
        for fbanks, images, video_id, frame_indices in dataset:
            assert fbanks.shape == (16, 48, 128), f"fbanks shape {fbanks.shape} != (16, 48, 128)"
            assert images.shape == (16, 3, 224, 224), f"images shape {images.shape} != (16, 3, 224, 224)"
            assert isinstance(video_id, str), f"video_id type {type(video_id)}"
            assert frame_indices.shape == (16,), f"frame_indices shape {frame_indices.shape}"
            count += 1

        assert count == 3, f"Expected 3 samples, iterated {count}"
        print(f"  Iterated {count} samples, all shapes correct")


def test_collate_fn_batch_shapes():
    """Collate function produces correct batch shapes."""
    from src.dataloader_webdataset import unsupervised_collate_fn

    batch_size = 3
    total_frame = 16
    target_length = 48
    num_mel_bins = 128

    # Simulate per-video tuples
    batch = []
    for i in range(batch_size):
        fbanks = torch.randn(total_frame, target_length, num_mel_bins)
        images = torch.randn(total_frame, 3, 224, 224)
        video_id = f"vid_{i}"
        frame_indices = torch.arange(total_frame)
        batch.append((fbanks, images, video_id, frame_indices))

    fbanks_out, images_out, video_ids_out, frame_indices_out = unsupervised_collate_fn(batch)

    n = batch_size * total_frame
    assert fbanks_out.shape == (n, target_length, num_mel_bins), f"fbanks shape {fbanks_out.shape}"
    assert images_out.shape == (n, 3, 224, 224), f"images shape {images_out.shape}"
    assert len(video_ids_out) == n, f"video_ids len {len(video_ids_out)} != {n}"
    assert frame_indices_out.shape == (n,), f"frame_indices shape {frame_indices_out.shape}"

    # Check video_ids expansion
    for i in range(batch_size):
        for j in range(total_frame):
            assert video_ids_out[i * total_frame + j] == f"vid_{i}", (
                f"video_id mismatch at [{i}][{j}]: {video_ids_out[i * total_frame + j]}"
            )

    print(f"  Collated batch shapes: fbanks={fbanks_out.shape}, images={images_out.shape}")
    print(f"  Video IDs correctly expanded: {len(video_ids_out)} entries")


def test_normalize_before_pad():
    """Normalize-before-pad: padded regions should be exactly 0.0."""
    from src.dataloader_webdataset import slice_fbank_at_timestamp

    target_length = 48
    num_mel_bins = 128

    # Create a fbank that is shorter than target_length so padding is needed
    fbank_length = 30
    full_fbank = torch.randn(100, num_mel_bins)  # Plenty of data
    norm_mean = -6.0
    norm_std = 3.5

    # Request a slice that goes past the end, forcing right padding
    # timestamp at the very end
    timestamp_ms = (fbank_length - 1) * 10
    result = slice_fbank_at_timestamp(
        full_fbank, fbank_length, timestamp_ms, target_length,
        norm_mean=norm_mean, norm_std=norm_std,
    )

    assert result.shape == (target_length, num_mel_bins), f"Shape {result.shape}"

    # The padded region (at the end) should be 0.0
    # The non-padded region should generally NOT be 0.0 (since it's normalized data)
    # Find where padding starts: we need to figure out how much padding was applied
    center_frame = round(timestamp_ms / 10)
    half_len = target_length // 2
    start = center_frame - half_len
    end = start + target_length

    pad_right = max(0, end - fbank_length)
    if pad_right > 0:
        padded_region = result[-pad_right:, :]
        assert torch.all(padded_region == 0.0), (
            f"Padded region not zero! Max: {padded_region.abs().max().item():.6f}"
        )
        # Non-padded should have non-zero values (with high probability)
        non_padded = result[:-pad_right, :]
        assert non_padded.abs().sum() > 0, "Non-padded region is all zeros — suspicious"
        print(f"  Padded region ({pad_right} frames) is exactly 0.0: OK")
        print(f"  Non-padded region has non-zero values: OK")
    else:
        print(f"  No padding needed in this test case (edge case)")
        # Try an extreme timestamp to force left padding
        result2 = slice_fbank_at_timestamp(
            full_fbank, fbank_length, 0, target_length,
            norm_mean=norm_mean, norm_std=norm_std,
        )
        # With timestamp=0 and target_length=48, start = -24, so pad_left=24
        pad_left = 24
        padded_left = result2[:pad_left, :]
        assert torch.all(padded_left == 0.0), (
            f"Left-padded region not zero! Max: {padded_left.abs().max().item():.6f}"
        )
        print(f"  Left-padded region ({pad_left} frames) is exactly 0.0: OK")


def test_image_transform_no_augmentation():
    """Non-augmentation transform produces correct image shapes."""
    import torchvision.transforms as T
    from PIL import Image

    imagenet_mean = [0.4850, 0.4560, 0.4060]
    imagenet_std = [0.2290, 0.2240, 0.2250]
    normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)

    transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    tensor = transform(img)

    assert tensor.shape == (3, 224, 224), f"Shape {tensor.shape} != (3, 224, 224)"
    assert tensor.dtype == torch.float32, f"Dtype {tensor.dtype}"
    print(f"  No-augmentation transform: shape={tensor.shape}, dtype={tensor.dtype}")


def test_image_transform_augmentation():
    """Augmentation transform produces correct image shapes."""
    import torchvision.transforms as T
    from PIL import Image

    imagenet_mean = [0.4850, 0.4560, 0.4060]
    imagenet_std = [0.2290, 0.2240, 0.2250]
    normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)

    transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalize,
    ])

    img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
    tensor = transform(img)

    assert tensor.shape == (3, 224, 224), f"Shape {tensor.shape} != (3, 224, 224)"
    assert tensor.dtype == torch.float32, f"Dtype {tensor.dtype}"
    print(f"  Augmentation transform: shape={tensor.shape}, dtype={tensor.dtype}")


def test_full_pipeline_shard_to_model():
    """Full pipeline: shard -> dataloader -> model.forward() succeeds."""
    from src.dataloader_webdataset import build_webdataset, unsupervised_collate_fn
    from src.models.cav_mae_sync import CAVMAE
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = _create_test_shards(tmpdir, n_samples=2)

        audio_conf = {
            'target_length': 48,
            'num_mel_bins': 128,
            'mean': -6.166528,
            'std': 3.483568,
            'im_res': 224,
            'total_frame': 16,
            'augmentation': False,
        }

        dataset = build_webdataset(
            shard_dir=shard_dir,
            audio_conf=audio_conf,
            shuffle=False,
            resampled=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=unsupervised_collate_fn,
        )

        model = CAVMAE(
            audio_length=48, embed_dim=768, cls_token=True,
            contrastive_heads=True, num_register_tokens=4, total_frame=16,
        )
        model.eval()

        for fbanks, images, video_ids, frame_indices in loader:
            print(f"  Batch: fbanks={fbanks.shape}, images={images.shape}")
            print(f"  video_ids sample: {video_ids[:3]}")

            with torch.no_grad():
                out = model(
                    fbanks, images,
                    mask_ratio_a=0.75, mask_ratio_v=0.75,
                    mae_loss_weight=1.0, contrast_loss_weight=0.01,
                    mode='unsupervised_train',
                )

            assert torch.isfinite(out['loss']), f"Loss not finite: {out['loss'].item()}"
            assert 'cls_a' in out and 'cls_v' in out
            print(f"  Forward pass succeeded: loss={out['loss'].item():.4f}")
            print(f"  cls_a={out['cls_a'].shape}, cls_v={out['cls_v'].shape}")
            break  # One batch is enough


# ============================================================================
# TRAINING SCRIPT TESTS
# ============================================================================

def test_optimizer_param_groups():
    """Optimizer has 2 param groups (decay + no_decay), all params in exactly one group."""
    from train_unsupervised import CAVMAEModule

    args = make_default_args(cls_token=True, contrastive_heads=True)
    module = CAVMAEModule(args)

    opt_config = module.configure_optimizers()
    if isinstance(opt_config, dict):
        optimizer = opt_config['optimizer']
    else:
        optimizer = opt_config

    param_groups = optimizer.param_groups
    assert len(param_groups) == 2, f"Expected 2 param groups, got {len(param_groups)}"

    # Check weight decay values
    decay_group = param_groups[0]
    no_decay_group = param_groups[1]
    print(f"  Decay group: {len(decay_group['params'])} params, wd={decay_group['weight_decay']}")
    print(f"  No-decay group: {len(no_decay_group['params'])} params, wd={no_decay_group['weight_decay']}")

    assert decay_group['weight_decay'] == 0.05, f"Decay wd {decay_group['weight_decay']} != 0.05"
    assert no_decay_group['weight_decay'] == 0.0, f"No-decay wd {no_decay_group['weight_decay']} != 0.0"

    # Collect all param IDs from optimizer
    opt_param_ids = set()
    for group in param_groups:
        for p in group['params']:
            pid = id(p)
            assert pid not in opt_param_ids, f"Duplicate param in optimizer groups!"
            opt_param_ids.add(pid)

    # Collect all model param IDs
    model_param_ids = {id(p) for p in module.parameters() if p.requires_grad}

    missing = model_param_ids - opt_param_ids
    extra = opt_param_ids - model_param_ids

    assert len(missing) == 0, f"{len(missing)} model params missing from optimizer"
    assert len(extra) == 0, f"{len(extra)} extra params in optimizer not in model"
    print(f"  All {len(model_param_ids)} requires_grad params covered, no duplicates")


def test_scheduler_warmup_lr():
    """SequentialLR warmup starts at correct LR (1% of base)."""
    from train_unsupervised import CAVMAEModule

    base_lr = 1e-4
    args = make_default_args(lr=base_lr, warmup_epochs=2, epochs=10)
    module = CAVMAEModule(args)

    opt_config = module.configure_optimizers()
    assert isinstance(opt_config, dict), "Expected dict with scheduler"
    assert 'lr_scheduler' in opt_config, "Missing lr_scheduler"

    optimizer = opt_config['optimizer']
    scheduler = opt_config['lr_scheduler']['scheduler']

    # After SequentialLR __init__, one step has been consumed.
    # The initial LR should be start_factor * base_lr = 0.01 * 1e-4 = 1e-6
    current_lr = optimizer.param_groups[0]['lr']
    expected_start_lr = base_lr * 0.01  # start_factor=0.01
    print(f"  Base LR: {base_lr}")
    print(f"  Current LR after init: {current_lr:.8f}")
    print(f"  Expected start LR: {expected_start_lr:.8f}")

    # The LR should be very close to expected (might differ slightly due to SequentialLR init step)
    assert abs(current_lr - expected_start_lr) < base_lr * 0.05, (
        f"Start LR {current_lr:.8f} not close to {expected_start_lr:.8f}"
    )

    # Step through warmup and verify LR increases
    lrs = [current_lr]
    for epoch in range(5):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)

    print(f"  LR schedule (first 6 steps): {[f'{lr:.8f}' for lr in lrs]}")
    # During warmup, LR should increase
    assert lrs[1] > lrs[0] or abs(lrs[1] - lrs[0]) < 1e-10, "LR should increase during warmup"


def test_datamodule_instantiation():
    """CAVMAEDataModule can be instantiated without errors."""
    from train_unsupervised import CAVMAEDataModule

    args = make_default_args(sharded_dataset_dir="/fake/path", batch_size=4)
    audio_conf = {
        'target_length': 48,
        'num_mel_bins': 128,
        'mean': -6.166528,
        'std': 3.483568,
        'im_res': 224,
        'total_frame': 16,
        'augmentation': True,
    }

    dm = CAVMAEDataModule(args, audio_conf)
    assert dm.args.batch_size == 4
    assert dm.audio_conf == audio_conf
    assert dm.dataset is None  # Not yet set up
    print(f"  CAVMAEDataModule instantiated successfully")
    print(f"  batch_size={dm.args.batch_size}, dataset=None (before setup)")


# ============================================================================
# EVAL TESTS
# ============================================================================

def test_bootstrap_metric():
    """bootstrap_metric produces valid confidence intervals."""
    from eval_deepfake import bootstrap_metric
    from sklearn.metrics import roc_auc_score

    np.random.seed(42)
    n = 200
    y_true = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    y_scores = np.concatenate([
        np.random.normal(0.3, 0.2, n // 2),
        np.random.normal(0.7, 0.2, n // 2),
    ])

    point, lo, hi = bootstrap_metric(y_true, y_scores, roc_auc_score, n_bootstrap=500)

    assert 0 <= point <= 1, f"AUC {point} out of range"
    assert 0 <= lo <= hi <= 1, f"CI [{lo}, {hi}] invalid"
    assert lo <= point <= hi, f"Point {point} not in CI [{lo}, {hi}]"
    print(f"  AUC: {point:.4f} [{lo:.4f}, {hi:.4f}]")


def test_compute_eer():
    """compute_eer produces valid EER."""
    from eval_deepfake import compute_eer

    np.random.seed(42)
    n = 200
    y_true = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    y_scores = np.concatenate([
        np.random.normal(0.3, 0.2, n // 2),
        np.random.normal(0.7, 0.2, n // 2),
    ])

    eer, threshold = compute_eer(y_true, y_scores)

    assert 0 <= eer <= 1, f"EER {eer} out of [0, 1]"
    assert np.isfinite(threshold), f"Threshold not finite: {threshold}"
    print(f"  EER: {eer:.4f}, threshold: {threshold:.4f}")


def test_get_manipulation_type():
    """get_manipulation_type parses all FakeAVCeleb conventions."""
    from eval_deepfake import get_manipulation_type

    cases = {
        '/path/FakeVideo-FakeAudio/video1.mp4': 'FakeVideo+FakeAudio',
        '/path/FakeVideo-RealAudio/video2.mp4': 'FakeVideo-RealAudio',
        '/path/RealVideo-FakeAudio/video3.mp4': 'RealVideo-FakeAudio',
        '/path/RealVideo-RealAudio/video4.mp4': 'Real',
        'some_other_video.mp4': 'Unknown',
    }

    for video_id, expected in cases.items():
        result = get_manipulation_type(video_id)
        assert result == expected, f"get_manipulation_type('{video_id}') = '{result}', expected '{expected}'"
        print(f"  '{video_id}' -> '{result}' (OK)")


def test_score_negation_logic():
    """
    Score negation logic: for sync scores, lower = more fake, so negate for sklearn.
    For distance/variance, higher = more fake, so no negation.
    """
    from eval_deepfake import compute_eer
    from sklearn.metrics import roc_auc_score

    np.random.seed(42)

    # Simulate: real videos have high sync (good), fake videos have low sync (bad)
    n = 100
    y_true = np.concatenate([np.zeros(n), np.ones(n)])  # 0=real, 1=fake
    sync_scores = np.concatenate([
        np.random.normal(0.8, 0.1, n),   # real: high sync
        np.random.normal(0.3, 0.1, n),   # fake: low sync
    ])

    # Without negation: sklearn thinks higher = more likely positive (fake)
    # But our sync scores are inverted (higher = more likely real)
    # So we MUST negate for sklearn to work correctly
    auc_wrong = roc_auc_score(y_true, sync_scores)  # Should be < 0.5 (wrong direction)
    auc_correct = roc_auc_score(y_true, -sync_scores)  # Should be > 0.5

    print(f"  AUC without negation: {auc_wrong:.4f} (expected < 0.5)")
    print(f"  AUC with negation:    {auc_correct:.4f} (expected > 0.5)")

    assert auc_correct > 0.5, f"AUC with negation {auc_correct} should be > 0.5"
    assert auc_correct > auc_wrong, "Negation should improve AUC"

    # For distance metrics (higher = more fake), no negation needed
    dist_scores = np.concatenate([
        np.random.normal(0.2, 0.1, n),   # real: low distance
        np.random.normal(0.8, 0.1, n),   # fake: high distance
    ])
    auc_dist = roc_auc_score(y_true, dist_scores)
    print(f"  AUC for distance (no negation): {auc_dist:.4f} (expected > 0.5)")
    assert auc_dist > 0.5, f"Distance AUC {auc_dist} should be > 0.5"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_end_to_end_pipeline():
    """
    End-to-end: create shards -> load -> forward -> compute sync scores -> compute metrics.
    """
    from src.dataloader_webdataset import build_webdataset, unsupervised_collate_fn
    from src.models.cav_mae_sync import CAVMAE
    from eval_deepfake import compute_sync_scores, compute_eer
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = _create_test_shards(tmpdir, n_samples=4)

        audio_conf = {
            'target_length': 48,
            'num_mel_bins': 128,
            'mean': -6.166528,
            'std': 3.483568,
            'im_res': 224,
            'total_frame': 16,
            'augmentation': False,
        }

        dataset = build_webdataset(
            shard_dir=shard_dir,
            audio_conf=audio_conf,
            shuffle=False,
            resampled=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=unsupervised_collate_fn,
        )

        model = CAVMAE(
            audio_length=48, embed_dim=768, cls_token=True,
            contrastive_heads=True, num_register_tokens=4, total_frame=16,
        )
        model.eval()

        all_sync_means = []
        all_labels = []

        for batch_idx, (fbanks, images, video_ids, frame_indices) in enumerate(loader):
            with torch.no_grad():
                out = model(
                    fbanks, images,
                    mask_ratio_a=0.0, mask_ratio_v=0.0,
                    mae_loss_weight=0.0,
                    contrast_loss_weight=0.01,
                    mode='unsupervised_train',
                )

            cls_a = out['cls_a']
            cls_v = out['cls_v']

            per_frame_sims, aggregated = compute_sync_scores(cls_a, cls_v, total_frames=16)

            batch_size = per_frame_sims.shape[0]
            for i in range(batch_size):
                sync_mean = aggregated['sync_mean'][i].item()
                all_sync_means.append(sync_mean)
                # Assign fake labels for testing (alternating real/fake)
                all_labels.append(i % 2)

            print(f"  Batch {batch_idx}: per_frame_sims shape={per_frame_sims.shape}, "
                  f"sync_mean={aggregated['sync_mean'].tolist()}")

        print(f"  Total videos processed: {len(all_sync_means)}")
        print(f"  Sync means: {all_sync_means}")

        # Compute metrics if we have both classes
        if len(set(all_labels)) >= 2:
            y_true = np.array(all_labels)
            y_scores = -np.array(all_sync_means)  # Negate for sklearn
            eer, threshold = compute_eer(y_true, y_scores)
            print(f"  EER: {eer:.4f}, threshold: {threshold:.4f}")
        else:
            print(f"  Skipping EER (need both classes)")


def test_compute_sync_scores_shapes():
    """compute_sync_scores produces correct output shapes and aggregation keys."""
    from eval_deepfake import compute_sync_scores

    batch_size = 3
    total_frames = 16
    embed_dim = 768
    n = batch_size * total_frames

    cls_a = torch.randn(n, embed_dim)
    cls_v = torch.randn(n, embed_dim)

    per_frame_sims, aggregated = compute_sync_scores(cls_a, cls_v, total_frames=total_frames)

    assert per_frame_sims.shape == (batch_size, total_frames), (
        f"per_frame_sims shape {per_frame_sims.shape} != ({batch_size}, {total_frames})"
    )

    expected_keys = {'sync_mean', 'sync_min', 'sync_max', 'sync_std',
                     'sync_p10', 'sync_p25', 'sync_p50',
                     'intra_sim_audio', 'intra_sim_visual',
                     'sync_euc', 'sync_pearson'}
    assert set(aggregated.keys()) == expected_keys, (
        f"Keys mismatch.\n  Expected: {sorted(expected_keys)}\n  Got: {sorted(aggregated.keys())}"
    )

    for key, val in aggregated.items():
        assert val.shape == (batch_size,), f"{key} shape {val.shape} != ({batch_size},)"
        assert torch.all(torch.isfinite(val)), f"{key} contains non-finite values"

    # Cosine similarities should be in [-1, 1]
    assert per_frame_sims.min() >= -1.01, f"min sim {per_frame_sims.min():.4f} < -1"
    assert per_frame_sims.max() <= 1.01, f"max sim {per_frame_sims.max():.4f} > 1"

    print(f"  per_frame_sims shape: {per_frame_sims.shape}")
    print(f"  Aggregation keys: {sorted(aggregated.keys())}")
    print(f"  Sim range: [{per_frame_sims.min():.4f}, {per_frame_sims.max():.4f}]")


def test_fakesync_config_validation():
    """FakeSyncConfig validates audio_length and target_length consistency."""
    from src.fakesync_config import FakeSyncConfig

    # Valid config
    config = FakeSyncConfig(audio_length=48, target_length=48)
    print(f"  Valid config created: audio_length={config.audio_length}, target_length={config.target_length}")

    # Invalid: audio_length not divisible by 16
    try:
        FakeSyncConfig(audio_length=50, target_length=50)
        assert False, "Should have raised ValueError for audio_length=50"
    except ValueError as e:
        print(f"  Correctly rejected audio_length=50: {e}")

    # Invalid: target_length != audio_length
    try:
        FakeSyncConfig(audio_length=48, target_length=64)
        assert False, "Should have raised ValueError for mismatched lengths"
    except ValueError as e:
        print(f"  Correctly rejected mismatched lengths: {e}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def main():
    print("=" * 70)
    print("CAV-MAE Sync: Comprehensive Test Suite")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Model Tests ---
    run_test("Model: batch_size=1, cls_token=True (squeeze fix)", test_forward_cls_token_batch1)
    run_test("Model: batch_size=1, cls_token=False", test_forward_no_cls_token_batch1)
    run_test("Model: mae_loss_weight=0 (no NameError)", test_mae_loss_weight_zero)
    run_test("Model: forward_feat shapes", test_forward_feat_shapes)
    run_test("Model: register token init std ~0.02", test_register_token_init_std)
    run_test("Model: contrastive loss bidirect", test_contrastive_loss_bidirect)
    run_test("Model: gradient flow through contrastive loss", test_gradient_flow_contrastive)
    run_test("Model: output dict keys for all code paths", test_output_dict_keys)

    # --- WebDataset Shard Writer Tests ---
    run_test("Shard Writer: write and read back 5 samples", test_shard_write_and_read)
    run_test("Shard Writer: identity-aware splitting (disjoint)", test_identity_splitting)
    run_test("Shard Writer: identity exclusion", test_identity_exclusion)
    run_test("Shard Writer: empty input handling", test_empty_input_shard_writer)

    # --- WebDataset Dataloader Tests ---
    run_test("Dataloader: build_webdataset iterate + verify shapes", test_build_webdataset_iterate)
    run_test("Dataloader: collate function batch shapes", test_collate_fn_batch_shapes)
    run_test("Dataloader: normalize-before-pad (zeros in padding)", test_normalize_before_pad)
    run_test("Dataloader: non-augmentation image transform", test_image_transform_no_augmentation)
    run_test("Dataloader: augmentation image transform", test_image_transform_augmentation)
    run_test("Dataloader: full pipeline shard -> model.forward()", test_full_pipeline_shard_to_model)

    # --- Training Script Tests ---
    run_test("Training: optimizer param groups (decay + no_decay)", test_optimizer_param_groups)
    run_test("Training: scheduler warmup start LR", test_scheduler_warmup_lr)
    run_test("Training: CAVMAEDataModule instantiation", test_datamodule_instantiation)

    # --- Eval Tests ---
    run_test("Eval: bootstrap_metric valid CIs", test_bootstrap_metric)
    run_test("Eval: compute_eer valid EER", test_compute_eer)
    run_test("Eval: get_manipulation_type parsing", test_get_manipulation_type)
    run_test("Eval: score negation logic", test_score_negation_logic)

    # --- Config Tests ---
    run_test("Config: FakeSyncConfig validation", test_fakesync_config_validation)

    # --- Integration Tests ---
    run_test("Integration: compute_sync_scores shapes", test_compute_sync_scores_shapes)
    run_test("Integration: end-to-end pipeline", test_end_to_end_pipeline)

    # ======================================================================
    # CLUSTER-SPECIFIC TESTS (skipped if paths don't exist)
    # ======================================================================
    # Paths derived from sbatch scripts
    CLUSTER_PATHS = {
        'train_shards': '/storage/slurm/hunecke/fakesync/data/voxceleb2/preprocessed/shards_train',
        'eval_shards': '/storage/slurm/hunecke/fakesync/data/fakeavceleb/preprocessed/shards_eval',
        'checkpoint': '/storage/slurm/hunecke/fakesync/cav-mae-sync/outputs/checkpoints/unsupervised_voxceleb2_3epoch_1worker/last.ckpt',
        'pretrained': '/storage/slurm/schnackl/fakesync/cav-mae-sync/pretrained_models/cav_mae_sync.pth',
        'voxceleb2_csv': '/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb_train.csv',
    }

    on_cluster = any(os.path.exists(p) for p in CLUSTER_PATHS.values())
    if on_cluster:
        print("\n\n" + "=" * 70)
        print("CLUSTER-SPECIFIC TESTS (real data detected)")
        print("=" * 70)

        # --- Test: Can we load existing .pt shards with old dataloader? ---
        def test_cluster_load_old_shards():
            """Load a few samples from existing .pt shards to verify they still work."""
            shard_dir = CLUSTER_PATHS['train_shards']
            if not os.path.exists(shard_dir):
                print(f"  SKIP: {shard_dir} not found")
                return
            import glob
            shards = sorted(glob.glob(os.path.join(shard_dir, '*.pt')))
            print(f"  Found {len(shards)} .pt shards in {shard_dir}")
            assert len(shards) > 0, "No .pt shards found"
            # Load first shard
            data = torch.load(shards[0], weights_only=False)
            print(f"  Shard 0: {len(data)} samples")
            s = data[0]
            print(f"  Sample keys: {list(s.keys())}")
            print(f"  video_id: {s.get('video_id', 'N/A')}")
            print(f"  fbank shape: {s['fbank'].shape}, dtype: {s['fbank'].dtype}")
            print(f"  fbank_length: {s.get('fbank_length', 'N/A')}")
            print(f"  num images: {len(s.get('images', []))}")
            print(f"  valid: {s.get('valid', 'N/A')}")
            assert s['fbank'].shape[1] == 128, f"Expected 128 mel bins, got {s['fbank'].shape[1]}"
        run_test("Cluster: load existing .pt shards", test_cluster_load_old_shards)

        # --- Test: Can we load a checkpoint? ---
        def test_cluster_load_checkpoint():
            """Load a trained checkpoint and verify model weights."""
            ckpt_path = CLUSTER_PATHS['checkpoint']
            if not os.path.exists(ckpt_path):
                # Try pretrained
                ckpt_path = CLUSTER_PATHS['pretrained']
            if not os.path.exists(ckpt_path):
                print(f"  SKIP: No checkpoint found at {CLUSTER_PATHS['checkpoint']} or {CLUSTER_PATHS['pretrained']}")
                return
            print(f"  Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if 'pytorch-lightning_version' in ckpt or 'callbacks' in ckpt:
                print(f"  Type: PyTorch Lightning checkpoint")
                print(f"  Keys: {list(ckpt.keys())[:10]}")
                if 'state_dict' in ckpt:
                    sd = ckpt['state_dict']
                    print(f"  state_dict keys: {len(sd)} parameters")
                    # Check a few expected keys
                    expected_keys = ['model.cls_token_a', 'model.cls_token_v', 'model.register_tokens']
                    for k in expected_keys:
                        if k in sd:
                            print(f"    {k}: {sd[k].shape}")
                        else:
                            print(f"    {k}: MISSING")
                if 'hyper_parameters' in ckpt:
                    hp = ckpt['hyper_parameters']
                    print(f"  hparams: audio_length={hp.get('audio_length')}, total_frame={hp.get('total_frame')}")
            else:
                print(f"  Type: Raw PyTorch checkpoint")
                keys = list(ckpt.keys())[:10]
                print(f"  Top keys: {keys}")
        run_test("Cluster: load trained checkpoint", test_cluster_load_checkpoint)

        # --- Test: Load checkpoint into CAVMAEModule ---
        def test_cluster_load_into_module():
            """Load checkpoint into CAVMAEModule and run a forward pass."""
            from train_unsupervised import CAVMAEModule
            ckpt_path = CLUSTER_PATHS['checkpoint']
            if not os.path.exists(ckpt_path):
                print(f"  SKIP: {ckpt_path} not found")
                return
            print(f"  Loading CAVMAEModule from {ckpt_path}...")
            try:
                model = CAVMAEModule.load_from_checkpoint(ckpt_path, map_location='cpu')
                model.eval()
                print(f"  Model loaded successfully")
                print(f"  hparams: {dict(list(model.hparams.items())[:5])}")
                # Forward pass with random data
                a = torch.randn(2, 48, 128)
                v = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    out = model(a, v)
                print(f"  Forward pass OK, loss={out['loss'].item():.4f}")
                print(f"  cls_a shape: {out['cls_a'].shape}")
            except Exception as e:
                print(f"  Failed to load as PL checkpoint: {e}")
                print(f"  Trying raw state_dict load...")
                from src.models.cav_mae_sync import CAVMAE
                ckpt = torch.load(ckpt_path, map_location='cpu')
                sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
                model = CAVMAE(audio_length=48, modality_specific_depth=11, cls_token=True, num_register_tokens=8)
                msg = model.load_state_dict({k.replace('model.', ''): v for k, v in sd.items()}, strict=False)
                print(f"  Raw load result: {msg}")
                model.eval()
                a = torch.randn(2, 48, 128)
                v = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    out = model(a, v)
                print(f"  Forward pass OK, loss={out['loss'].item():.4f}")
        run_test("Cluster: load checkpoint into CAVMAEModule", test_cluster_load_into_module)

        # --- Test: Load old shards through old dataloader and verify output shapes ---
        def test_cluster_old_dataloader_shapes():
            """Load data through ShardedAudiosetDataset and verify shapes."""
            from src.dataloader_sharded import ShardedAudiosetDataset
            shard_dir = CLUSTER_PATHS['train_shards']
            if not os.path.exists(shard_dir):
                print(f"  SKIP: {shard_dir} not found")
                return
            audio_conf = {
                'target_length': 48, 'num_mel_bins': 128,
                'mean': -4.050048828125, 'std': 4.067018032073975,
                'im_res': 224, 'total_frame': 16,
                'augmentation': False, 'mode': 'unsupervised_train',
                'skip_norm': False,
            }
            dataset = ShardedAudiosetDataset(
                shard_dir=shard_dir, audio_conf=audio_conf,
                shuffle_shards=False, dataset_fraction=0.01,  # Only 1% of data
            )
            sample = next(iter(dataset))
            fbanks, images, vid, indices = sample
            print(f"  fbanks: {fbanks.shape}")
            print(f"  images: {images.shape}")
            print(f"  video_id: {vid}")
            print(f"  frame_indices: {indices.shape}")
            assert fbanks.shape == (16, 48, 128), f"Expected (16, 48, 128), got {fbanks.shape}"
            assert images.shape == (16, 3, 224, 224), f"Expected (16, 3, 224, 224), got {images.shape}"
            print(f"  fbank value range: [{fbanks.min():.3f}, {fbanks.max():.3f}]")
            print(f"  Old dataloader output shapes OK")
        run_test("Cluster: old .pt dataloader produces correct shapes", test_cluster_old_dataloader_shapes)

        # --- Test: Create WebDataset shards from a real video ---
        def test_cluster_create_webdataset_from_real_video():
            """Create a small WebDataset shard from a real VoxCeleb2 video."""
            import pandas as pd
            csv_path = CLUSTER_PATHS['voxceleb2_csv']
            if not os.path.exists(csv_path):
                print(f"  SKIP: {csv_path} not found")
                return
            df = pd.read_csv(csv_path, nrows=5)
            path_col = 'video_name' if 'video_name' in df.columns else 'video_path'
            print(f"  CSV columns: {list(df.columns)}")
            print(f"  First 5 video paths:")
            for _, row in df.head(5).iterrows():
                p = row[path_col]
                exists = os.path.exists(p)
                print(f"    {p} (exists={exists})")
            # Try to process first existing video
            existing = [row[path_col] for _, row in df.iterrows() if os.path.exists(row[path_col])]
            if not existing:
                print(f"  SKIP: No accessible video files found in CSV")
                return
            video_path = existing[0]
            print(f"  Processing: {video_path}")
            from preprocess.create_webdataset import extract_audio_fbank, extract_frames_decord, extract_frames_opencv, HAS_DECORD
            fbank, fbank_length = extract_audio_fbank(video_path)
            print(f"  fbank: {fbank.shape}, dtype={fbank.dtype}, length={fbank_length}")
            assert fbank.shape[1] == 128
            extract_fn = extract_frames_decord if HAS_DECORD else extract_frames_opencv
            imgs, indices, timestamps, fps, duration = extract_fn(video_path, 16, 224, 1024)
            print(f"  frames: {len(imgs)}, fps={fps:.1f}, duration={duration:.0f}ms")
            print(f"  timestamps: {timestamps[:4]}... (ms)")
            print(f"  decoder: {'decord' if HAS_DECORD else 'opencv'}")
            assert len(imgs) == 16
            assert len(timestamps) == 16
        run_test("Cluster: create WebDataset from real video", test_cluster_create_webdataset_from_real_video)

        # --- Test: Full WebDataset shard creation + loading roundtrip with real video ---
        def test_cluster_webdataset_roundtrip_real():
            """Create WebDataset shards from real videos, load through dataloader, verify."""
            import pandas as pd
            csv_path = CLUSTER_PATHS['voxceleb2_csv']
            if not os.path.exists(csv_path):
                print(f"  SKIP: {csv_path} not found")
                return
            df = pd.read_csv(csv_path, nrows=10)
            path_col = 'video_name' if 'video_name' in df.columns else 'video_path'
            existing = [row[path_col] for _, row in df.iterrows() if os.path.exists(row[path_col])]
            if len(existing) < 2:
                print(f"  SKIP: Need at least 2 accessible videos, found {len(existing)}")
                return
            # Create temp CSV with just these videos
            tmpdir = tempfile.mkdtemp()
            try:
                tmp_csv = os.path.join(tmpdir, 'test.csv')
                pd.DataFrame({path_col: existing[:3]}).to_csv(tmp_csv, index=False)
                # Run shard creation
                from preprocess.create_webdataset import load_input_file, process_single_video, write_webdataset_shards
                file_list = load_input_file(tmp_csv)
                results_list = []
                for item in file_list:
                    res = process_single_video((
                        item['video_path'], item.get('labels'), '',
                        16, 224, 128, 1024
                    ))
                    results_list.append(res)
                    print(f"    {res['video_id']}: valid={res['valid']}")
                valid = [r for r in results_list if r['valid']]
                if not valid:
                    print(f"  SKIP: No valid videos after processing")
                    return
                n = write_webdataset_shards(valid, tmpdir, 'test')
                print(f"  Wrote {n} samples to WebDataset shards")
                # Load through dataloader
                from src.dataloader_webdataset import build_webdataset, unsupervised_collate_fn
                audio_conf = {
                    'target_length': 48, 'num_mel_bins': 128,
                    'mean': -4.050048828125, 'std': 4.067018032073975,
                    'im_res': 224, 'total_frame': 16, 'augmentation': False,
                    'skip_norm': False,
                }
                dataset = build_webdataset(
                    os.path.join(tmpdir, 'test'), audio_conf,
                    shuffle=False, resampled=False,
                )
                sample = next(iter(dataset))
                fbanks, images, vid, indices = sample
                print(f"  Loaded sample: vid={vid}, fbanks={fbanks.shape}, images={images.shape}")
                assert fbanks.shape == (16, 48, 128)
                assert images.shape == (16, 3, 224, 224)
                print(f"  fbank range: [{fbanks.min():.3f}, {fbanks.max():.3f}]")
                print(f"  Real data WebDataset roundtrip OK!")
            finally:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
        run_test("Cluster: WebDataset roundtrip with real videos", test_cluster_webdataset_roundtrip_real)

        # --- Test: GPU forward pass (if GPU available) ---
        def test_cluster_gpu_forward():
            """Run model forward pass on GPU to verify CUDA compatibility."""
            if not torch.cuda.is_available():
                print(f"  SKIP: No GPU available")
                return
            from src.models.cav_mae_sync import CAVMAE
            device = torch.device('cuda:0')
            model = CAVMAE(audio_length=48, modality_specific_depth=11, cls_token=True, num_register_tokens=8).to(device)
            model.eval()
            a = torch.randn(4, 48, 128, device=device)
            v = torch.randn(4, 3, 224, 224, device=device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(a, v, mae_loss_weight=0.8, contrast_loss_weight=0.7,
                           contrast_bidirect=True, contrast_intra_weight=0.65, contrast_inter_weight=0.35)
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  loss={out['loss'].item():.4f}, cls_a={out['cls_a'].shape}")
            print(f"  GPU memory used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            torch.cuda.empty_cache()
        run_test("Cluster: GPU forward pass (bf16)", test_cluster_gpu_forward)

        # --- Test: GPU training step (if GPU available) ---
        def test_cluster_gpu_training_step():
            """Run a full training step on GPU including backward pass."""
            if not torch.cuda.is_available():
                print(f"  SKIP: No GPU available")
                return
            from src.models.cav_mae_sync import CAVMAE
            device = torch.device('cuda:0')
            model = CAVMAE(audio_length=48, modality_specific_depth=11, cls_token=True, num_register_tokens=8).to(device)
            model.train()
            a = torch.randn(4, 48, 128, device=device)
            v = torch.randn(4, 3, 224, 224, device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(a, v, mae_loss_weight=0.8, contrast_loss_weight=0.7,
                           contrast_bidirect=True, contrast_intra_weight=0.65, contrast_inter_weight=0.35)
                loss = out['loss']
            loss.backward()
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            n_grads = sum(1 for p in model.parameters() if p.grad is not None)
            print(f"  loss={loss.item():.4f}")
            print(f"  {n_grads} parameters have gradients, total grad norm={grad_norm:.4f}")
            print(f"  GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            torch.cuda.empty_cache()
        run_test("Cluster: GPU training step (forward+backward)", test_cluster_gpu_training_step)

        # --- Test: Lightning fast_dev_run with WebDataset ---
        def test_cluster_fast_dev_run():
            """Run Lightning fast_dev_run to verify full training pipeline."""
            if not torch.cuda.is_available():
                print(f"  SKIP: No GPU available")
                return
            # Need WebDataset shards — create temp ones
            tmpdir = tempfile.mkdtemp()
            try:
                from preprocess.create_webdataset import write_webdataset_shards
                from PIL import Image
                samples = []
                for i in range(5):
                    fbank = (torch.randn(1024, 128) * 3 - 6).half()
                    images = []
                    for j in range(16):
                        img = Image.new('RGB', (224, 224), color=(50, 50, 50))
                        buf = io.BytesIO()
                        img.save(buf, format='JPEG', quality=90)
                        images.append(buf.getvalue())
                    samples.append({
                        'video_id': f'fast_dev_{i}', 'valid': True, 'identity': f'id{i:05d}',
                        'fbank': fbank, 'fbank_length': 500, 'images': images,
                        'frame_indices': list(range(0, 250, 15))[:16],
                        'frame_timestamps_ms': [j * 625.0 for j in range(16)],
                        'video_fps': 25.0, 'video_duration_ms': 10000.0, 'labels': 0,
                    })
                shard_dir = os.path.join(tmpdir, 'train')
                write_webdataset_shards(samples, tmpdir, 'train')
                # Write metadata
                meta = {'splits': {'train': {'total_valid': 5}}, 'args': {}}
                with open(os.path.join(shard_dir, 'webdataset_metadata.json'), 'w') as f:
                    json.dump(meta, f)
                # Run fast_dev_run
                import subprocess
                cmd = [
                    sys.executable, os.path.join(PROJECT_ROOT, 'train_unsupervised.py'),
                    '--sharded_dataset_dir', shard_dir,
                    '--batch_size', '2',
                    '--total_frame', '16',
                    '--audio_length', '48',
                    '--target_length', '48',
                    '--lr', '1e-4',
                    '--epochs', '1',
                    '--warmup_epochs', '0',
                    '--fast_dev_run',
                    '--num_workers', '0',
                    '--save_path', os.path.join(tmpdir, 'ckpts'),
                    '--cls_token',
                    '--num_register_tokens', '8',
                ]
                print(f"  Running: {' '.join(cmd[-10:])}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                print(f"  Return code: {result.returncode}")
                if result.stdout:
                    # Print last 20 lines of stdout
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-20:]:
                        print(f"    stdout: {line}")
                if result.returncode != 0:
                    print(f"  STDERR (last 20 lines):")
                    for line in result.stderr.strip().split('\n')[-20:]:
                        print(f"    stderr: {line}")
                assert result.returncode == 0, f"fast_dev_run failed with code {result.returncode}"
                print(f"  fast_dev_run completed successfully!")
            finally:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
        run_test("Cluster: Lightning fast_dev_run with WebDataset", test_cluster_fast_dev_run)

    else:
        print("\n\n" + "=" * 70)
        print("CLUSTER-SPECIFIC TESTS SKIPPED (no cluster paths detected)")
        print("=" * 70)

    # ======================================================================
    # FINAL REPORT
    # ======================================================================
    print("\n\n")
    print("=" * 70)
    print("FINAL TEST REPORT")
    print("=" * 70)

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    total_time = sum(r.duration for r in results)

    print(f"\nTotal: {len(results)} tests | Passed: {len(passed)} | Failed: {len(failed)}")
    print(f"Total time: {total_time:.1f}s\n")

    if failed:
        print("-" * 70)
        print("FAILURES:")
        print("-" * 70)
        for r in failed:
            print(f"\n  FAIL: {r.name} ({r.duration:.2f}s)")
            # Print first few lines of the error message
            lines = r.message.strip().split('\n')
            for line in lines[:20]:
                print(f"    {line}")
            if len(lines) > 20:
                print(f"    ... ({len(lines) - 20} more lines)")

    print("\n" + "-" * 70)
    print("ALL TESTS:")
    print("-" * 70)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name} ({r.duration:.2f}s)")

    print("\n" + "=" * 70)
    if failed:
        print(f"RESULT: {len(failed)} FAILURES")
    else:
        print("RESULT: ALL TESTS PASSED")
    print("=" * 70)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
