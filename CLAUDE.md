# ResNet CIFAR-10 Reproduction — repo guide

Reproduce He et al. 2015 (arXiv:1512.03385) CIFAR-10 Table 6, then extend to CIFAR-100
and pre-activation ResNet-56. Full spec + task list in `PLAN.md`.

Repo: https://github.com/AmroAbujabal/resnet-cifar-repro

## Status (2026-08-08)

- **Published, public, MIT.** README + one-click Colab notebook.
- **22 tests pass locally on CPU**; the 11 data tests need CIFAR, which this network cannot fetch,
  so the full suite (33) runs on cloud GPU — see below.
- **Param counts match Table 6 exactly** (ResNet-20 269,722 … ResNet-110 1,727,962).
- **T1 confirmed green on Colab (2026-08-05).** Locally the CIFAR mirror is unreachable (Toronto
  301→dead `cave` HTTP host; the HTTPS cave endpoint runs ~23 KB/s and never completes — resume
  corrupts, single-shot times out). Environment issue, not a code defect, as the Colab run proved.
- **Phase 2 COMPLETE (2026-08-08), 3 seeds each:** ResNet-20 **8.39 ± 0.31%** (paper 8.75%, Δ−0.36)
  and ResNet-56 **7.45 ± 0.69%** (paper 6.97%, Δ+0.48). Both inside ±0.5% — the DoD is met.
  ResNet-56 clears it by 0.02% only because seed 2 came in at 8.22%; see the README caveats.
- `results.csv` is **tracked**, not ignored: it is the reported source of truth. The Colab run
  writes it to `MyDrive/resnet-repro/results.csv` and the Kaggle run to `/kaggle/working/`, so
  seeds survive a reclaimed session. It gained a `train_error_pct` column for Phase 3 (PLAN.md
  wants train error per cell); the six Phase 2 rows are **empty** there, not zero — those runs
  never measured it. `train.py` asserts the on-disk header matches the row it is about to append,
  because `DictWriter` would otherwise silently misalign every value against an older schema.
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

- **Locally (no dataset needed):** the model/config/metric/train/eval suites —
  `.venv/bin/python -m pytest tests/ --ignore=tests/test_data.py -q`
- **Full suite incl. T1 data tests:** run on Colab (or after placing a valid
  `data/cifar-10-python.tar.gz`, md5 `c58f30108f718f92721af3b95e74349a`) — `python -m pytest -q`.
  Do NOT run the full suite locally against the live mirror; the T1 download hangs.

## Definition of done (Phase 2)

ResNet-20 and ResNet-56 within **±0.5% absolute** of paper error (≤9.25%, ≤7.47%),
mean over ≥3 seeds.

## Phase 3 (code landed, runs pending)

The 2×2 is {original, pre-act} ResNet-56 × {CIFAR-10, CIFAR-100}. `resnet56` × CIFAR-10 is
already done, so three configs remain: `preact56`, `resnet56_c100`, `preact56_c100`. Everything
except `name` / `dataset` / `num_classes` / `preact` is copied from `resnet56.yaml` and held fixed.

- Pre-act = `resnet(n, preact=True)` (He 2016 Fig 4e, full pre-activation, clean identity).
  **Param count is exactly identical to the original** — the head BN(64) costs +96, and the two
  dimension-changing blocks give back −96 by normalizing `in_c` instead of `out_c`.
  Dimension-changing blocks follow the reference's `both_preact`: the shared BN-ReLU is applied
  **before** the branch split, so the Option-A shortcut subsamples the pre-activated signal, not
  the raw signed input. A test pins this — it is easy to get wrong and invisible in the loss curve.
- CIFAR-100 reuses the whole pipeline via `dataset="cifar100"`; the per-pixel mean is recomputed
  from its own train split (a test asserts it differs from CIFAR-10's).
- Config `name` is the `model` column in `results.csv` and the key the notebooks skip on, so the
  four cells never collide. `tests/test_configs.py` builds every config before it costs GPU time.

## Phase 3 progress

- ✅ `resnet56` × CIFAR-10 — **7.45 ± 0.69%** (Phase 2).
- ✅ `preact56` × CIFAR-10 — **7.22 ± 0.29%** (Kaggle version 2, 2026-08-11, commit `1ec7995`).
  0.23% better than the original, which is **smaller than either std → within noise**, not a win.
- ✅ `resnet56_c100` — **29.75 ± 0.42%** (29.33 / 30.16 / 29.77; Kaggle version 3, 2026-08-12,
  commit `f60ac13`). Train error 0.5–0.6%, so the net fully fits 50k images over 100 classes.
- 🔄 `preact56_c100` — running as version 4 (launched 2026-08-12 ~19:50 local, ETA ~03:00 Aug 13).

CIFAR-100 has **no paper baseline** here (He 2015 Table 6 is CIFAR-10 only), so those two cells are
an internal pre-act-vs-original comparison at fixed budget, not a reproduction claim.

**Write-up framing (decided 2026-08-12): the 2×2 reports "no detectable difference at depth 56"** —
a null result stated as a finding. Do not add seeds to chase significance; report every Δ against
the combined std and label it within noise. If a cell comes back with a Δ *larger* than the combined
std, that is new information — report it, don't force the null.

## Next step

Run the remaining configs **one per Kaggle saved version** (~7.2 h each: ~1.1 h of pytest and CIFAR
downloads, then 3 seeds × ~2.0 h). Set `PHASE3` in the Phase 3 cell of
`notebooks/reproduce_kaggle.ipynb`, then `cd notebooks && ../.venv/bin/kaggle kernels push -p .`
(this **launches immediately — there is no dry-run**). Pull the output with `kernels output`, commit
`results.csv`, push, repeat. Two rules that cost real GPU hours when broken:

- **Push `results.csv` to GitHub _before_ launching the next run** — the notebook seeds its results
  file from the cloned repo, so an unpushed row is a row the next run neither skips nor carries.
- **Never put two configs in one version** — a run killed at the session limit discards
  `/kaggle/working` entirely.
