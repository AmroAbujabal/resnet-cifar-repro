"""CIFAR-10 data pipeline (He et al. 2015, Sec 4.2).

Train view: 4-px pad -> random 32x32 crop + random horizontal flip.
Test/val view: original 32x32, untouched.
Per-pixel mean subtraction, mean computed from the TRAIN split only.
Downsample/color-aug decisions per PLAN.md: pad+crop+flip only, no color aug.
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_N_TRAIN = 50000


def _split_indices(seed: int, val_size: int, n: int = _N_TRAIN):
    """Deterministic seeded train/val partition of range(n)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    if val_size == 0:
        return perm, []
    return perm[: n - val_size], perm[n - val_size :]


def raw_train_split(root: str, val_size: int = 5000, seed: int = 0, download: bool = True):
    """The un-augmented train split as a float tensor (N,3,32,32) in [0,1]."""
    base = datasets.CIFAR10(root, train=True, download=download)
    data = torch.from_numpy(base.data).permute(0, 3, 1, 2).float() / 255.0  # RGB CHW
    train_idx, _ = _split_indices(seed, val_size)
    return data[train_idx]


def compute_per_pixel_mean(root: str, val_size: int = 5000, seed: int = 0, download: bool = True):
    """Per-pixel mean image (3,32,32) over the train split only."""
    return raw_train_split(root, val_size, seed, download).mean(dim=0)


class _SubtractPerPixelMean:
    def __init__(self, mean):
        self.mean = mean  # (3,32,32)

    def __call__(self, x):
        return x - self.mean


def _transforms(mean, train: bool):
    steps = []
    if train:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), _SubtractPerPixelMean(mean)]
    return transforms.Compose(steps)


def get_datasets(root: str, val_size: int = 5000, seed: int = 0, download: bool = True,
                 full_train: bool = False):
    """train (augmented), val, test (both un-augmented), normalized by train mean."""
    mean = compute_per_pixel_mean(root, 0 if full_train else val_size, seed, download)
    train_tf = _transforms(mean, train=True)
    eval_tf = _transforms(mean, train=False)

    train_base = datasets.CIFAR10(root, train=True, download=download, transform=train_tf)
    eval_base = datasets.CIFAR10(root, train=True, download=download, transform=eval_tf)
    test_ds = datasets.CIFAR10(root, train=False, download=download, transform=eval_tf)

    train_idx, val_idx = _split_indices(seed, val_size)
    train_ds = Subset(train_base, list(range(_N_TRAIN)) if full_train else train_idx)
    val_ds = Subset(eval_base, val_idx)
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def get_loaders(root: str, batch_size: int = 128, val_size: int = 5000, seed: int = 0,
                download: bool = True, full_train: bool = False, num_workers: int = 0):
    ds = get_datasets(root, val_size, seed, download, full_train)
    return {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True),
        "val": DataLoader(ds["val"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers),
        "test": DataLoader(ds["test"], batch_size=batch_size, shuffle=False,
                           num_workers=num_workers),
    }
