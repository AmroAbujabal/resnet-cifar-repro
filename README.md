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

**Training (GPU):** open the [Colab notebook](notebooks/reproduce_colab.ipynb) or the
[Kaggle notebook](notebooks/reproduce_kaggle.ipynb) — each runs the full suite and trains
ResNet-20/56, skipping any seed already recorded in `results.csv` — or locally:

```bash
python scripts/train.py --config configs/resnet56.yaml --seed 0 --device cuda
```

## Reproduced vs. paper

Trained on T4 GPUs (Colab, then Kaggle) with the schedule above, full 50k train split, single-view
test evaluation. All numbers come from [`results.csv`](results.csv); none are hand-copied.

|     Model | Paper (Table 6) |   Reproduced (mean ± std, 3 seeds) | Δ          | Per-seed error      | Wall clock |
| --------: | :-------------: | ---------------------------------: | :--------- | :------------------ | ---------: |
| ResNet-20 |      8.75%      | **8.39 ± 0.31%**                   | **−0.36%** | 8.30 / 8.13 / 8.73  |    140 min |
| ResNet-56 |      6.97%      | **7.45 ± 0.69%**                   | **+0.48%** | 6.91 / 7.21 / 8.22  |    378 min |

**Both models reproduce within the ±0.5% target on the mean of 3 seeds** (seeds 0, 1, 2), which is
this project's definition of done.

Two caveats worth stating plainly:

- **ResNet-56 clears the bar by 0.02%.** Its mean is dragged up by seed 2 at 8.22% — 1.3% worse than
  seed 0 and well outside the other two. Seeds 0 and 1 (6.91%, 7.21%) straddle the paper's 6.97%
  closely; on those alone the match would look much tighter. One unlucky trajectory out of three is
  consistent with the deeper net being harder to optimise, but n=3 cannot separate that from noise.
- **ResNet-56's std (0.69%) is more than double ResNet-20's (0.31%)**, so its interval genuinely
  overlaps the paper's number — the point estimate is not the whole story.

More seeds would tighten both intervals; 3 is the minimum this repo committed to, not a claim that
3 is enough to characterise the tail.

## Extension: pre-activation vs. original (in progress)

A controlled 2×2 — **{original, pre-activation} ResNet-56** × **{CIFAR-10, CIFAR-100}** — with seeds,
schedule, augmentation and **parameter budget** held fixed (the pre-activation variant has _exactly_
the same 853,018 parameters, not approximately). Three of the four cells have run:

|      Model | Dataset   | Test error (mean ± std, 3 seeds) | Per-seed error        |
| ---------: | :-------- | -------------------------------: | :-------------------- |
| ResNet-56  | CIFAR-10  |                   7.45 ± 0.69%   | 6.91 / 7.21 / 8.22    |
| pre-act 56 | CIFAR-10  |               **7.22 ± 0.29%**   | 7.08 / 7.03 / 7.55    |
| ResNet-56  | CIFAR-100 |                  29.75 ± 0.42%   | 29.33 / 30.16 / 29.77 |
| pre-act 56 | CIFAR-100 |                        _pending_ | —                     |

Pre-activation is **0.23% better on the mean — smaller than either std, so this is within noise**, not
a demonstrated improvement. What is more suggestive is the spread: the original's seed 2 blew out to
8.22%, while the pre-activation runs stayed inside a 0.5% band. He et al. (2016) attribute exactly
that to the clean identity path easing optimisation, but n=3 at one depth cannot establish it. At
depth 56 the paper itself reports the two orderings as near-identical; the gap it documents opens at
110+ layers.

**CIFAR-100 has no paper baseline** — He et al. (2015) Table 6 is CIFAR-10 only, so the CIFAR-100 row
is not a reproduction claim. It is the same network and schedule pointed at a 100-way problem, and it
exists to give the pre-activation comparison a second dataset at fixed budget. The original ResNet-56
lands at **29.75 ± 0.42%** there, with train error near zero (0.5–0.6%) — the model fully fits 50k
images across 100 classes and the remaining error is generalisation, not underfitting. The
pre-activation cell is still to run.

## Status

- ✅ Paper extraction + gap decisions (`PLAN.md`)
- ✅ Model, metric, training loop, evaluation — built test-first
- ✅ Parameter counts verified against Table 6 (above)
- ✅ **Full suite green on real CIFAR-10 + CIFAR-100: 36 passing** (Kaggle T4) — including the data-pipeline tests
- ✅ **Phase 2 done: ResNet-20 and ResNet-56 reproduced within ±0.5% over 3 seeds** (table above)
- 🔄 **Phase 3 (2×2 extension): 3 of 4 cells run** — both CIFAR-10 cells + original ResNet-56 on
  CIFAR-100; pre-act ResNet-56 on CIFAR-100 is the last one

## References

- He, Zhang, Ren, Sun. _Deep Residual Learning for Image Recognition._ arXiv:1512.03385 (2015).
- He, Zhang, Ren, Sun. _Identity Mappings in Deep Residual Networks._ arXiv:1603.05027 (2016).
