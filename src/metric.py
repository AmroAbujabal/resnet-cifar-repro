"""Top-1 classification error % (He et al. 2015 CIFAR metric)."""
import torch


def top1_error(logits, targets) -> float:
    """100 * (misclassified / N), argmax over the class dimension."""
    preds = logits.argmax(dim=1)
    wrong = (preds != targets).sum().item()
    return 100.0 * wrong / targets.numel()


class ErrorMeter:
    """Accumulates top-1 error over batches for a single-view test pass."""

    def __init__(self):
        self.wrong = 0
        self.total = 0

    def update(self, logits, targets):
        self.wrong += (logits.argmax(dim=1) != targets).sum().item()
        self.total += targets.numel()

    def error(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.wrong / self.total
