# Reproducing ResNet on CIFAR-10

A from-scratch reproduction of the CIFAR-10 experiments in **[Deep Residual Learning for Image
Recognition](https://arxiv.org/abs/1512.03385)** (He, Zhang, Ren & Sun, 2015) — rebuilt from the
paper alone, under test-driven development, then extended into a controlled study of residual
learning across depth, architecture, and dataset.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmroAbujabal/resnet-cifar-repro/blob/main/notebooks/reproduce_colab.ipynb)

---

## What this is

The goal is not to _use_ a ResNet — it is to **rebuild one from the specification** and land within
**±0.5% absolute test error** of the paper's Table 6, averaged over ≥3 seeds. Every architectural and
training detail was extracted from Section 4.2 and reconstructed: the 6n+2 layer family, parameter-free
Option-A identity shortcuts, per-pixel mean normalization, pad-crop-flip augmentation, and the exact
SGD schedule.

The model, metric, and training loop are built **test-first** — each has a failing test written and
watched fail before a line of implementation. The model builder asserts layer counts and parameter
counts _against the paper_ before training is ever attempted.

## Verified: parameter counts match the paper exactly

`resnet(n)` reproduces the Table 6 model family to the parameter:

|      Model |  n  | Layers (6n+2) |    Params | Paper | Reported error |
| ---------: | :-: | :-----------: | --------: | :---: | :------------: |
|  ResNet-20 |  3  |      20       |   269,722 | 0.27M |     8.75%      |
|  ResNet-32 |  5  |      32       |   464,154 | 0.46M |     7.51%      |
|  ResNet-44 |  7  |      44       |   658,586 | 0.66M |     7.17%      |
|  ResNet-56 |  9  |      56       |   853,018 | 0.85M |     6.97%      |
| ResNet-110 | 18  |      110      | 1,727,962 | 1.7M  |     6.43%      |

## Approach

- **Read the paper as a spec.** Every unspecified-but-reproduction-relevant gap (seed set, 45k/5k vs.
  full-50k training, downsample mechanism, weight-decay scope, color augmentation) is documented in
  [`PLAN.md`](PLAN.md) with the explicit decision taken for each.
- **Test-driven throughout.** Data pipeline, top-1 metric, model construction, and the training loop
  each have tests. The single-batch overfit gate must drive one batch to 0% error before any full run.
- **Honest reporting.** ≥3 seeds per configuration, results reported as mean ± std. Any difference
  smaller than the combined std is flagged as _within noise_. All runs append to one `results.csv` —
  no hand-copied numbers.
- **No shortcuts on the model.** Built from `nn.Conv2d` up; `torchvision.models.resnet` (the ImageNet
  stem) is deliberately **not** used, since it does not match the CIFAR architecture.

## Architecture (CIFAR variant)

```
3x3 conv, 16  ->  BN  ->  ReLU
  stage 1: n BasicBlocks, 16 ch, 32x32
  stage 2: n BasicBlocks, 32 ch, 16x16   (first block: stride-2 downsample)
  stage 3: n BasicBlocks, 64 ch,  8x8    (first block: stride-2 downsample)
global average pool  ->  64-d  ->  FC -> softmax
```

Shortcuts are **Option A**: on a dimension increase, the identity is subsampled (stride 2) and the new
channels are zero-padded — adding **no parameters and no extra compute**. He/MSRA initialization, BN
after every conv, no dropout.

## Repository layout

```
src/     data.py    CIFAR pipeline: split, per-pixel mean, train/test transforms
         model.py   resnet(n): BasicBlock + Option-A shortcut + He init
         metric.py  top-1 error % and a streaming ErrorMeter
         train.py   single training step (SGD + cross-entropy)
         eval.py    single-view test-set evaluation
tests/              one test module per component (TDD)
configs/            resnet20.yaml, resnet56.yaml — the He et al. schedule
scripts/train.py    full training driver -> results.csv
notebooks/          reproduce_colab.ipynb — one-click GPU reproduction
PLAN.md             paper extraction + task list + documented gap decisions
```

## Running it

**Tests (CPU, seconds):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

**Training (GPU):** open the [Colab notebook](notebooks/reproduce_colab.ipynb) — it runs the full
suite and trains ResNet-20/56 — or locally:

```bash
python scripts/train.py --config configs/resnet56.yaml --seed 0 --device cuda
```

## Status

- ✅ Paper extraction + gap decisions (`PLAN.md`)
- ✅ Model, metric, training loop, evaluation — built test-first, **15 tests passing** (CPU)
- ✅ Parameter counts verified against Table 6 (above)
- 🔜 Full training runs on GPU (Colab) — `results.csv` and the reproduced-vs-paper table land here next

## Extensions (planned)

A controlled 2×2 study, holding seeds / schedule / budget fixed:
**{original, pre-activation ResNet-56}** × **{CIFAR-10, CIFAR-100}** — isolating the effect of the
[pre-activation ordering](https://arxiv.org/abs/1603.05027) (He et al., 2016) and of class count.

## References

- He, Zhang, Ren, Sun. _Deep Residual Learning for Image Recognition._ arXiv:1512.03385 (2015).
- He, Zhang, Ren, Sun. _Identity Mappings in Deep Residual Networks._ arXiv:1603.05027 (2016).
