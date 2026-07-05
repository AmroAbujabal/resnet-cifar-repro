"""T1 — CIFAR-10 data loader + augmentation.

Spec (PLAN.md Phase 2 T1 + He et al. 2015 Sec 4.2):
  - train view: 4-px pad -> random 32x32 crop, + random horizontal flip
  - test/val view: original 32x32, untouched (no crop, no flip, no TTA)
  - per-pixel mean subtraction, mean computed from the TRAIN split only
  - schedule-tuning split = 45k train / 5k val / 10k test, train & val disjoint
"""
import torch
from src.data import get_datasets, get_loaders, compute_per_pixel_mean, raw_train_split

ROOT = "data"


def test_split_sizes_45k_5k_10k():
    ds = get_datasets(root=ROOT, val_size=5000, seed=0)
    assert len(ds["train"]) == 45000
    assert len(ds["val"]) == 5000
    assert len(ds["test"]) == 10000


def test_train_val_indices_disjoint_and_cover_50k():
    ds = get_datasets(root=ROOT, val_size=5000, seed=0)
    train_idx = set(ds["train"].indices)
    val_idx = set(ds["val"].indices)
    assert train_idx.isdisjoint(val_idx)
    assert len(train_idx | val_idx) == 50000


def test_split_is_deterministic_given_seed():
    a = get_datasets(root=ROOT, val_size=5000, seed=0)["val"].indices
    b = get_datasets(root=ROOT, val_size=5000, seed=0)["val"].indices
    assert list(a) == list(b)


def test_train_batch_shape_is_128x3x32x32():
    loaders = get_loaders(root=ROOT, batch_size=128, val_size=5000, seed=0)
    x, y = next(iter(loaders["train"]))
    assert x.shape == (128, 3, 32, 32)
    assert y.shape == (128,)


def test_test_view_is_untouched_and_deterministic():
    # no random augmentation on the test view: same index -> identical tensor
    ds = get_datasets(root=ROOT, val_size=5000, seed=0)
    x0, _ = ds["test"][7]
    x1, _ = ds["test"][7]
    assert torch.equal(x0, x1)


def test_train_view_is_augmented():
    # random crop+flip on the train view: re-reading the same index re-samples
    # the transform, so the views should almost always differ
    ds = get_datasets(root=ROOT, val_size=5000, seed=0)
    views = [ds["train"][0][0] for _ in range(12)]
    distinct = sum(1 for v in views if not torch.equal(v, views[0]))
    assert distinct >= 8


def test_per_pixel_mean_shape():
    mean = compute_per_pixel_mean(root=ROOT, val_size=5000, seed=0)
    assert mean.shape == (3, 32, 32)


def test_per_pixel_mean_centers_training_data():
    # subtracting the per-pixel mean centres the (un-augmented) train split at ~0
    mean = compute_per_pixel_mean(root=ROOT, val_size=5000, seed=0)
    raw = raw_train_split(root=ROOT, val_size=5000, seed=0)  # (45000,3,32,32) in [0,1]
    centered = (raw - mean).mean().abs().item()
    assert centered < 1e-5


def test_mean_uses_train_only_not_val():
    # mean over the 45k train split must differ from mean over the full 50k
    # (guards against val/test leakage into the statistic)
    train_mean = compute_per_pixel_mean(root=ROOT, val_size=5000, seed=0)
    full_mean = compute_per_pixel_mean(root=ROOT, val_size=0, seed=0)
    assert not torch.allclose(train_mean, full_mean)
