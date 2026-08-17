"""Training step and run-level seeding for the CIFAR ResNet (SGD + cross-entropy)."""
import os
import random

import numpy as np
import torch
import torch.nn as nn

_default_criterion = nn.CrossEntropyLoss()


def set_seed(seed: int):
    """Seed every RNG a run touches and ask for deterministic kernels.

    A seed label is only worth writing into results.csv if it identifies the run.
    cudnn.benchmark is the usual reason two identical runs diverge on a GPU: the
    autotuner may pick a different convolution algorithm per launch, and the
    algorithms do not agree bit for bit. warn_only=True is deliberate -- an op
    with no deterministic implementation should warn, not kill a two-hour run.
    Must be called before the first CUDA op: CUBLAS_WORKSPACE_CONFIG is only read
    when cuBLAS initialises.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_step(model, batch, optimizer, criterion=None) -> float:
    """One optimization step on a batch; returns the scalar loss."""
    criterion = criterion or _default_criterion
    model.train()
    x, y = batch
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()
