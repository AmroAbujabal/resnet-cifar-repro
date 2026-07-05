"""T2 — top-1 classification error % (PLAN.md metric: 100 * misclassified / N)."""
import torch
from src.metric import top1_error, ErrorMeter


def _logits_for(preds, num_classes=10):
    """Build logits whose argmax equals the given predicted class per row."""
    z = torch.zeros(len(preds), num_classes)
    for i, p in enumerate(preds):
        z[i, p] = 1.0
    return z


def test_all_correct_is_zero():
    logits = _logits_for([0, 1, 2, 3])
    targets = torch.tensor([0, 1, 2, 3])
    assert top1_error(logits, targets) == 0.0


def test_all_wrong_is_100():
    logits = _logits_for([1, 2, 3, 4])
    targets = torch.tensor([0, 1, 2, 3])
    assert top1_error(logits, targets) == 100.0


def test_half_wrong_is_50():
    logits = _logits_for([0, 9, 2, 9])
    targets = torch.tensor([0, 1, 2, 3])
    assert top1_error(logits, targets) == 50.0


def test_uses_argmax_of_logits():
    # row 0: class 2 has the highest logit -> predicts 2 (correct)
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.0]])
    targets = torch.tensor([2])
    assert top1_error(logits, targets) == 0.0


def test_meter_accumulates_across_batches():
    m = ErrorMeter()
    m.update(_logits_for([0, 1]), torch.tensor([0, 1]))   # 2 correct
    m.update(_logits_for([9, 9]), torch.tensor([0, 1]))   # 2 wrong
    assert m.error() == 50.0
    assert m.total == 4


def test_meter_empty_is_zero():
    assert ErrorMeter().error() == 0.0
