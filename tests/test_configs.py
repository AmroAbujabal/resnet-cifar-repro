"""Every config must build the model it claims, before it costs GPU hours.

A typo in a Phase 3 config is otherwise only discovered ~2h into a Kaggle run.
"""
import glob

import torch
import yaml

from src.model import resnet

CONFIGS = sorted(glob.glob("configs/*.yaml"))


def test_configs_exist():
    assert CONFIGS


def test_head_width_matches_the_dataset():
    # The likely typo is copying resnet56.yaml, setting dataset: cifar100 and forgetting
    # num_classes: 100. That builds fine and only dies on the first batch, as an opaque
    # device-side assert, hours into a saved Kaggle version.
    for path in CONFIGS:
        cfg = yaml.safe_load(open(path))
        dataset = cfg.get("dataset", "cifar10")
        assert dataset in ("cifar10", "cifar100"), path
        assert cfg.get("num_classes", 10) == (100 if dataset == "cifar100" else 10), path


def test_every_config_builds():
    for path in CONFIGS:
        cfg = yaml.safe_load(open(path))
        classes = cfg.get("num_classes", 10)
        model = resnet(cfg["n"], num_classes=classes, preact=cfg.get("preact", False))
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, classes), path


def test_config_names_are_unique():
    # `name` is the `model` column in results.csv and the key the notebooks use to
    # skip finished (model, seed) pairs -- two configs sharing one would collide.
    names = [yaml.safe_load(open(p))["name"] for p in CONFIGS]
    assert len(set(names)) == len(names), names
