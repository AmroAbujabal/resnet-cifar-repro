"""T3 — CIFAR ResNet builder resnet(n).

Spec (He et al. 2015 Sec 4.2, Table 6):
  - weighted layers = 6n+2 (1 stem conv + 6n stage convs + 1 FC)
  - param counts (~): n=3->0.27M, n=5->0.46M, n=7->0.66M, n=9->0.85M, n=18->1.7M
  - Option-A shortcuts (identity + zero-pad) add NO parameters -> no 1x1 projection convs
  - output shape (B, 10)
"""
import torch
import torch.nn as nn
from src.model import resnet


def _count_weighted_layers(model):
    return sum(isinstance(m, (nn.Conv2d, nn.Linear)) for m in model.modules())


def _param_count(model):
    return sum(p.numel() for p in model.parameters())


def test_layer_count_is_6n_plus_2():
    for n in (3, 5, 7, 9, 18):
        assert _count_weighted_layers(resnet(n)) == 6 * n + 2, f"n={n}"


def test_param_counts_match_paper_table():
    # Table 6 values, checked at the precision the paper quotes them (2 dp, except 1.7M at 1 dp).
    expected = {3: (0.27, 2), 5: (0.46, 2), 7: (0.66, 2), 9: (0.85, 2), 18: (1.7, 1)}
    for n, (value, dp) in expected.items():
        got = _param_count(resnet(n)) / 1e6
        assert round(got, dp) == value, f"n={n}: got {got:.4f}M, expected {value}M"


def test_output_shape_is_batch_by_10():
    out = resnet(3)(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_option_a_has_no_projection_convs():
    # Option B would insert 1x1 conv shortcuts (extra params); Option A must not.
    for m in resnet(9).modules():
        if isinstance(m, nn.Conv2d):
            assert m.kernel_size == (3, 3), "found a non-3x3 conv (Option-B projection?)"


def test_num_classes_is_configurable():
    out = resnet(3, num_classes=100)(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 100)
