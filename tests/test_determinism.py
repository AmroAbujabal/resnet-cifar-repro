"""Same seed, same numbers -- or the seed column in results.csv means nothing.

Written after the fact: the Phase 2 runs were seeded but not made deterministic,
so seed 2 of ResNet-56 cannot be re-derived from its label alone. This pins the
property going forward. It runs on CPU, where determinism is easy; the T4 caveats
are in CLAUDE.md.
"""
from src.train import set_seed, train_step

STEPS = 5


def losses(harness, seed: int):
    """Loss at each of the first STEPS steps. The model, the batch and the optimizer
    are all built after set_seed, so all three are downstream of it."""
    set_seed(seed)
    model, x, y, opt = harness(8)
    return [train_step(model, (x, y), opt) for _ in range(STEPS)]


def test_same_seed_gives_identical_loss_at_step_n(harness):
    a, b = losses(harness, 0), losses(harness, 0)
    assert a[-1] == b[-1], f"step {STEPS} diverged: {a[-1]!r} vs {b[-1]!r}"
    assert a == b, f"{a} vs {b}"


def test_different_seeds_diverge(harness):
    # Without this, a loss that was constant for some other reason would let the
    # test above pass while proving nothing.
    assert losses(harness, 0)[-1] != losses(harness, 1)[-1]
