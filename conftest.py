import os
import sys

_root = os.path.dirname(__file__)
sys.path.insert(0, _root)
# scripts/ too, so tests can import build_site the same way they import src.
sys.path.insert(0, os.path.join(_root, "scripts"))

import pytest  # noqa: E402
import torch  # noqa: E402

from src.model import resnet  # noqa: E402


@pytest.fixture
def harness():
    """A ResNet-20, a synthetic batch and an SGD optimizer -- the setup three tests
    across two files were each writing out.

    It deliberately does NOT seed. The caller seeds first, because the determinism
    test's whole claim is that everything built after its seeding call is decided by
    it, and a fixture that seeded itself would quietly move that boundary.
    """
    def build(n=16, lr=0.1):
        model = resnet(3)
        x, y = torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,))
        return model, x, y, torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    return build
