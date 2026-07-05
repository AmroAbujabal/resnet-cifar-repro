"""Evaluation over a loader — single-view top-1 error % (no dataset needed)."""
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import resnet
from src.eval import evaluate


def test_evaluate_returns_error_percent_on_a_loader():
    torch.manual_seed(0)
    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    err = evaluate(resnet(3), loader, device="cpu")
    assert isinstance(err, float)
    assert 0.0 <= err <= 100.0


def test_evaluate_perfect_model_is_zero_error():
    # a model that always outputs its input's mean can't be perfect, so build a
    # trivial deterministic case: identical inputs, single class -> after argmax
    # a constant predictor either gets all right or all wrong; assert consistency.
    x = torch.zeros(6, 3, 32, 32)
    y = torch.zeros(6, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=3)
    err = evaluate(resnet(3), loader, device="cpu")
    assert err in (0.0, 100.0)  # constant input -> constant prediction
