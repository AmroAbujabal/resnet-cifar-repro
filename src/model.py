"""CIFAR ResNet (He et al. 2015, Sec 4.2): 6n+2 layers, Option-A shortcuts.

Stem 3x3/16 -> 3 stages of n BasicBlocks over feature sizes {32,16,8} with
{16,32,64} filters -> global avg pool -> FC. First block of stages 2 and 3
downsamples (stride 2). Shortcuts are identity + zero-pad (Option A, no params).

`preact=True` swaps in the full pre-activation block of He et al. 2016 at the
same depth and (exactly) the same parameter count -- Phase 3's comparison arm.
"""
import torch.nn as nn
import torch.nn.functional as F


class _OptionAShortcut(nn.Module):
    """Identity shortcut for dim increase: stride-2 subsample + zero-pad channels. No params."""

    def __init__(self, stride: int, pad_channels: int):
        super().__init__()
        self.stride = stride
        self.pad = pad_channels

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.pad > 0:
            top = self.pad // 2
            x = F.pad(x, (0, 0, 0, 0, top, self.pad - top))
        return x


def _shortcut(in_c: int, out_c: int, stride: int) -> nn.Module:
    """Option A everywhere: zero-pad on a dim increase, plain identity otherwise."""
    if stride != 1 or in_c != out_c:
        return _OptionAShortcut(stride, out_c - in_c)
    return nn.Identity()


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = _shortcut(in_c, out_c, stride)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class PreActBasicBlock(nn.Module):
    """Full pre-activation block (He et al. 2016, Fig 4e): BN-ReLU-conv twice.

    Where dimensions match, nothing sits on the shortcut path, so the identity is clean
    (no post-add ReLU). Where they change, this follows the reference's `both_preact`:
    the shared BN-ReLU is applied *before* the branch split, so the Option-A shortcut
    subsamples the pre-activated signal, not the raw signed input.
    """

    def __init__(self, in_c: int, out_c: int, stride: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.shortcut = _shortcut(in_c, out_c, stride)
        self.clean_identity = isinstance(self.shortcut, nn.Identity)

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(self.conv1(pre))))
        return out + (x if self.clean_identity else self.shortcut(pre))


class CifarResNet(nn.Module):
    def __init__(self, n: int, num_classes: int = 10, preact: bool = False):
        super().__init__()
        self.preact = preact
        block = PreActBasicBlock if preact else BasicBlock
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        # Pre-activation moves the stem's BN-ReLU into the first block and needs a
        # final BN-ReLU before pooling instead. The budget still matches exactly: this
        # BN(64) costs +96 params, and the two dimension-changing blocks give back
        # -32 -64 by normalizing in_c rather than out_c.
        if preact:
            self.bn_last = nn.BatchNorm2d(64)
        else:
            self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(block, 16, 16, n, stride=1)
        self.stage2 = self._make_stage(block, 16, 32, n, stride=2)
        self.stage3 = self._make_stage(block, 32, 64, n, stride=2)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(block, in_c, out_c, n, stride):
        blocks = [block(in_c, out_c, stride)]
        blocks += [block(out_c, out_c, 1) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        if not self.preact:
            out = F.relu(self.bn1(out))
        out = self.stage3(self.stage2(self.stage1(out)))
        if self.preact:
            out = F.relu(self.bn_last(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet(n: int, num_classes: int = 10, preact: bool = False) -> CifarResNet:
    return CifarResNet(n, num_classes, preact)
