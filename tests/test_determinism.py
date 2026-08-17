"""Same seed, same numbers -- or the seed column in results.csv means nothing.

Written after the fact: the Phase 2 runs were seeded but not made deterministic,
so seed 2 of ResNet-56 cannot be re-derived from its label alone. This pins the
property going forward. It runs on CPU, where determinism is easy; the T4 caveats
are in CLAUDE.md.
"""
import torch

from src.model import resnet
from src.train import set_seed, train_step

STEPS = 5


def losses(seed: int):
    """Loss at each of the first STEPS steps, everything downstream of set_seed."""
    set_seed(seed)
    model = resnet(3)
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return [train_step(model, (x, y), opt) for _ in range(STEPS)]


def test_same_seed_gives_identical_loss_at_step_n():
    a, b = losses(0), losses(0)
    assert a[-1] == b[-1], f"step {STEPS} diverged: {a[-1]!r} vs {b[-1]!r}"
    assert a == b, f"{a} vs {b}"


def test_different_seeds_diverge():
    # Without this, a loss that was constant for some other reason would let the
    # test above pass while proving nothing.
    assert losses(0)[-1] != losses(1)[-1]
