"""Single-view test-set evaluation -> top-1 error %."""
import torch

from src.metric import ErrorMeter


@torch.no_grad()
def evaluate(model, loader, device="cpu") -> float:
    """Top-1 error % over a loader, single original-image view (no TTA)."""
    model.eval()
    model.to(device)
    meter = ErrorMeter()
    for x, y in loader:
        logits = model(x.to(device))
        meter.update(logits.cpu(), y)
    return meter.error()
