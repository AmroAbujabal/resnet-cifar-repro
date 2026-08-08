# ResNet CIFAR-10 Reproduction — repo guide

Reproduce He et al. 2015 (arXiv:1512.03385) CIFAR-10 Table 6, then extend to CIFAR-100
and pre-activation ResNet-56. Full spec + task list in `PLAN.md`.

Repo: https://github.com/AmroAbujabal/resnet-cifar-repro

## Status (2026-08-08)

- **Published, public, MIT.** README + one-click Colab notebook.
- **24 tests passing on Colab GPU** (438s), including T1 on real CIFAR-10. 15 of those run on CPU
  locally; the 9 T1 data tests need CIFAR, which this network cannot fetch — see below.
- **Param counts match Table 6 exactly** (ResNet-20 269,722 … ResNet-110 1,727,962).
- **T1 confirmed green on Colab (2026-08-05).** Locally the CIFAR mirror is unreachable (Toronto
  301→dead `cave` HTTP host; the HTTPS cave endpoint runs ~23 KB/s and never completes — resume
  corrupts, single-shot times out). Environment issue, not a code defect, as the Colab run proved.
- **Phase 2 COMPLETE (2026-08-08), 3 seeds each:** ResNet-20 **8.39 ± 0.31%** (paper 8.75%, Δ−0.36)
  and ResNet-56 **7.45 ± 0.69%** (paper 6.97%, Δ+0.48). Both inside ±0.5% — the DoD is met.
  ResNet-56 clears it by 0.02% only because seed 2 came in at 8.22%; see the README caveats.
- `results.csv` is **tracked**, not ignored: it is the reported source of truth. The Colab run
  writes it to `MyDrive/resnet-repro/results.csv` and the Kaggle run to `/kaggle/working/`, so
  seeds survive a reclaimed session.
- **Kaggle is the runner** (`notebooks/reproduce_kaggle.ipynb`, 30 GPU-h/week, Save & Run All is
  headless). Colab's free GPU quota was exhausted and is not worth fighting.

## Environment

- **This machine (Intel i5, 2 cores, no GPU) is for code + tests only.** CPU training is infeasible.
- **Full training (T5) runs on cloud GPU** (Colab T4) via `notebooks/reproduce_colab.ipynb`.
- Interpreter: `/usr/local/bin/python3.11`. Project venv: `.venv/` (gitignored).
- torch pinned to 2.2.2 (last x86-macOS wheel); see `requirements.txt`.

## Confirmed decisions (2026-07-03)

- Final models train on **full 50k** (schedule tuned on 45k/5k val).
- Second dataset = **CIFAR-100**; comparison model = **pre-activation ResNet-56** (He 2016).
- Seeds **{0,1,2}**; **Option-A** downsample (strided subsample + channel zero-pad);
  weight decay on **all** params; **no color aug** (pad+crop+flip only).

## Workflow

- **TDD, strictly.** Write test → watch it fail → implement → pass. Validate cheap before expensive.
- Reference (mirror, do NOT copy, do NOT pip-install): `akamaster/pytorch_resnet_cifar10` (Option A).
- **Never** use `torchvision.models.resnet` — that's the ImageNet stem, won't match CIFAR numbers.
- All run results go to one `results.csv` — no hand-copied numbers.

## Running tests

- **Locally (no dataset needed):** the model/metric/train/eval suites —
  `.venv/bin/python -m pytest tests/test_metric.py tests/test_model.py tests/test_train.py tests/test_eval.py -q`
- **Full suite incl. T1 data tests:** run on Colab (or after placing a valid
  `data/cifar-10-python.tar.gz`, md5 `c58f30108f718f92721af3b95e74349a`) — `python -m pytest -q`.
  Do NOT run the full suite locally against the live mirror; the T1 download hangs.

## Definition of done (Phase 2)

ResNet-20 and ResNet-56 within **±0.5% absolute** of paper error (≤9.25%, ≤7.47%),
mean over ≥3 seeds.

## Next step

Phase 2 is validated, so Phase 3 is unblocked: pre-activation ResNet-56 × {CIFAR-10, CIFAR-100},
TDD as usual, run via `notebooks/reproduce_kaggle.ipynb`. Budget ~2.1 h/seed for a 56-layer run;
keep each Kaggle version under the session limit (split models across versions if tight).
