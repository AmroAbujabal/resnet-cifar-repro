"""CIFAR-10 / CIFAR-100 data pipeline (He et al. 2015, Sec 4.2).

Train view: 4-px pad -> random 32x32 crop + random horizontal flip.
Test/val view: original 32x32, untouched.
Per-pixel mean subtraction, mean computed from the TRAIN split only.
Downsample/color-aug decisions per PLAN.md: pad+crop+flip only, no color aug.
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_N_TRAIN = 50000
# CIFAR-100 is the same 50k/10k of 32x32x3 images, only the label space differs,
# so every split/augmentation/mean rule below applies unchanged (PLAN.md Phase 3A).
_DATASETS = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


def _split_indices(seed: int, val_size: int, n: int = _N_TRAIN):
    """Deterministic seeded train/val partition of range(n)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    if val_size == 0:
        return perm, []
    return perm[: n - val_size], perm[n - val_size :]


def raw_train_split(root: str, val_size: int = 5000, seed: int = 0, download: bool = True,
                    dataset: str = "cifar10"):
    """The un-augmented train split as a float tensor (N,3,32,32) in [0,1]."""
    base = _DATASETS[dataset](root, train=True, download=download)
    data = torch.from_numpy(base.data).permute(0, 3, 1, 2).float() / 255.0  # RGB CHW
    train_idx, _ = _split_indices(seed, val_size)
    return data[train_idx]


def compute_per_pixel_mean(root: str, val_size: int = 5000, seed: int = 0, download: bool = True,
                           dataset: str = "cifar10"):
    """Per-pixel mean image (3,32,32) over the train split only."""
    return raw_train_split(root, val_size, seed, download, dataset).mean(dim=0)


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
                 full_train: bool = False, dataset: str = "cifar10"):
    """train (augmented), val, test (both un-augmented), normalized by train mean."""
    mean = compute_per_pixel_mean(root, 0 if full_train else val_size, seed, download, dataset)
    train_tf = _transforms(mean, train=True)
    eval_tf = _transforms(mean, train=False)

    cls = _DATASETS[dataset]
    train_base = cls(root, train=True, download=download, transform=train_tf)
    eval_base = cls(root, train=True, download=download, transform=eval_tf)
    test_ds = cls(root, train=False, download=download, transform=eval_tf)

    train_idx, val_idx = _split_indices(seed, val_size)
    idx = list(range(_N_TRAIN)) if full_train else train_idx
    # train_eval: the same images as train, through the eval transform -- train error is
    # reported on the un-augmented view (PLAN.md Phase 3C wants it per cell).
    return {
        "train": Subset(train_base, idx),
        "train_eval": Subset(eval_base, idx),
        "val": Subset(eval_base, val_idx),
        "test": test_ds,
    }


def get_loaders(root: str, batch_size: int = 128, val_size: int = 5000, seed: int = 0,
                download: bool = True, full_train: bool = False, num_workers: int = 0,
                dataset: str = "cifar10"):
    ds = get_datasets(root, val_size, seed, download, full_train, dataset)
    return {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True),
        "train_eval": DataLoader(ds["train_eval"], batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers),
        "val": DataLoader(ds["val"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers),
        "test": DataLoader(ds["test"], batch_size=batch_size, shuffle=False,
                           num_workers=num_workers),
    }
