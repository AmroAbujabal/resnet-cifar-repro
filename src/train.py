"""Training step for the CIFAR ResNet (SGD + cross-entropy)."""
import torch.nn as nn

_default_criterion = nn.CrossEntropyLoss()


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
