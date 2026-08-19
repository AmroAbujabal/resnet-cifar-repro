"""T4 — single-batch overfit gate (PLAN.md: loss -> ~0 on one batch).

A correct model + loss + optimizer must be able to memorize a single batch.
Run before any full training. Uses synthetic data so it needs no dataset.
"""
import torch
from src.metric import top1_error
from src.train import train_step


def test_train_step_reduces_loss(harness):
    torch.manual_seed(0)
    model, x, y, opt = harness(16)
    first = train_step(model, (x, y), opt)
    for _ in range(20):
        last = train_step(model, (x, y), opt)
    assert last < first


def test_single_batch_overfits_to_zero_error(harness):
    torch.manual_seed(0)
    model, x, y, opt = harness(32)
    last = None
    for _ in range(150):
        last = train_step(model, (x, y), opt)
    assert last < 0.05, f"final loss {last:.4f} did not collapse"
    model.train()  # batch-stat BN, consistent with how it was trained
    assert top1_error(model(x), y) == 0.0
