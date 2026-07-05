"""CIFAR ResNet (He et al. 2015, Sec 4.2): 6n+2 layers, Option-A shortcuts.

Stem 3x3/16 -> 3 stages of n BasicBlocks over feature sizes {32,16,8} with
{16,32,64} filters -> global avg pool -> FC. First block of stages 2 and 3
downsamples (stride 2). Shortcuts are identity + zero-pad (Option A, no params).
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


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = _OptionAShortcut(stride, out_c - in_c)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, n: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, 16, n, stride=1)
        self.stage2 = self._make_stage(16, 32, n, stride=2)
        self.stage3 = self._make_stage(32, 64, n, stride=2)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_c, out_c, n, stride):
        blocks = [BasicBlock(in_c, out_c, stride)]
        blocks += [BasicBlock(out_c, out_c, 1) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage3(self.stage2(self.stage1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet(n: int, num_classes: int = 10) -> CifarResNet:
    return CifarResNet(n, num_classes)
