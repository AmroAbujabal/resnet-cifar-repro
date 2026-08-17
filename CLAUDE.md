# ResNet CIFAR-10 Reproduction — repo guide

Reproduce He et al. 2015 (arXiv:1512.03385) CIFAR-10 Table 6, then extend to CIFAR-100
and pre-activation ResNet-56. Full spec + task list in `PLAN.md`.

Repo: https://github.com/AmroAbujabal/resnet-cifar-repro

## Status (2026-08-08)

- **Published, public, MIT.** README + one-click Colab notebook.
- **27 tests pass locally on CPU** (2026-08-16); the 12 data tests need CIFAR, which this network
  cannot fetch, so the full suite (39) runs on cloud GPU — see below.
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

- Final models train on **full 50k** (schedule tuned on 45k/5k val). **Verified against the PDF
  on 2026-08-16 and it matches the paper**: Sec 4.2 says "we present experiments trained on the
  training set and evaluated on the test set" over "50k training images", and the 45k/5k split
  appears exactly once, as the source of one hyperparameter — training terminates at 64k
  iterations, "which is determined on a 45k/5k train/val split". `train_split: 50000` in every
  config drives `full_train=True` in `scripts/train.py`, so this work trains on the same 50k the
  paper did. The repro numbers carry no held-out-data advantage; the Method section says so.
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
- ✅ `preact56_c100` — **29.97 ± 0.37%** (29.96 / 30.35 / 29.61; Kaggle version 4, 2026-08-13).

CIFAR-100 has **no paper baseline** here (He 2015 Table 6 is CIFAR-10 only), so those two cells are
an internal pre-act-vs-original comparison at fixed budget, not a reproduction claim.

**PHASE 3 COMPLETE (2026-08-13). The finding is a null: no detectable difference at depth 56.**
Δ = −0.23% on CIFAR-10 (combined std 0.74) and **+0.22% on CIFAR-100** (combined std 0.56) — each
~⅓ of the combined std, and **opposite in sign**, which is the strongest form the null can take.
Matches He 2016, whose gap opens at 1001 layers, not 56. Pre-act has the smaller std on both
datasets; that stays an observation at n=3, not a claim. No extra seeds — chasing significance would
be fitting the experiment to a desired answer.

## Determinism (2026-08-16) — what is and is not pinned on a T4

`src.train.set_seed(seed)` is the one entry point, called by `scripts/train.py` before anything
touches CUDA. It pins python `random`, numpy, torch CPU and all CUDA devices; sets
`cudnn.deterministic = True` and `cudnn.benchmark = False` (the autotuner picking a different
convolution algorithm per launch is the usual reason two identical GPU runs diverge); sets
`torch.use_deterministic_algorithms(True, warn_only=True)` — `warn_only` on purpose, an op with
no deterministic kernel should print a warning, not kill a two-hour run at iteration 60k; and
sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which is only read when cuBLAS initialises, hence the
early call. DataLoader workers: torch seeds each worker's own generator from the parent seed and
that is what `RandomCrop` / `RandomHorizontalFlip` draw from, so the augmentation stream was
already deterministic; `_seed_worker` in `src/data.py` additionally seeds python/numpy per worker
so a future transform that reaches for either cannot silently break it.

**What this does NOT give you:**

- **Bit-identical agreement with the Phase 2 / Phase 3 runs.** Those ran with none of the above,
  so their conv algorithms were autotuner-chosen. Turning determinism on changes kernel selection,
  which changes the numerics from the first step. A rerun of `resnet56` seed 2 is a **new draw
  from the same seed label, not a replay** — it will not land on 8.22% except by coincidence, and
  that is itself the reportable finding about the original runs.
- Portability. Same seed on a different GPU, a different torch/cuDNN build, or a different
  `num_workers` gives a different run. Determinism here means _this_ setup repeats itself.
- GPU coverage from the test suite. `tests/test_determinism.py` runs on CPU, where determinism is
  cheap; it pins the seeding logic, not the cuDNN flags. Only a GPU run exercises those.

## Logs and the site build (2026-08-16)

- Every run writes `logs/<name>_seed<seed>.csv` — `iter,train_error_pct,test_error_pct` at each
  eval interval, flushed per row, **committed**. Phase 2/3 kept no logs, which is why Table 2's
  original × CIFAR-10 train error reads "not measured" — the one cell whose behaviour is worth
  explaining. `logs/preact56_c100_seed{0,1,2}.csv` are recovered from the Kaggle version-4 output
  and have an empty train column: those evaluations only measured test error.
- Train error is now measured at **every** eval interval, not once at the end. It costs one
  50k-image forward pass per interval (~8 per run, ≈2% of wall time) and it replaces the duplicate
  end-of-run evaluation the loop used to do, so a run is no slower than before.
- `scripts/build_site.py` writes every results-derived number into `site/index.html`: the
  `data-stat="KEY"` spans in prose and tables, and the figure data block between the `GENERATED`
  markers. **Never hand-edit a number in the page or inside those markers.** Run the script after
  any append; `--check` (wired into `tests/test_site.py`) fails if the page has drifted. It
  compares values, not bytes, because a formatter reflows the file.
- A duplicate `(model, seed)` in `results.csv` makes `build_site.py` **stop**. A rerun needs an
  explicit rule for which row the published aggregate uses, and that is a judgement call about
  what is being claimed, not something to default.
- **The rule, decided 2026-08-16:** a rerun gets its own config `name` (`resnet56_rerun`), so it
  never collides and never silently joins a pre-registered aggregate. Table 1, the Phase 2 mean
  and s.d., and the 2×2 keep meaning what they meant. The rerun supplies the train-error column
  and the Section 3 discussion only.

## Phase 4 (launched 2026-08-16)

`resnet56_rerun` = `resnet56` with a different name, seed 2 only, under the deterministic setup
with per-interval logging. It exists because the seed 2 excursion (8.22%, 1.31 points above seed 0) has no train-error curve behind it, so "deeper nets are harder to optimise" and "generalisation
wobble" are indistinguishable in the published write-up. What the outcome means:

- **~8.2% with elevated train error** → optimisation failure, supports the Section 3 reading.
- **~8.2% with normal train error** → generalisation wobble; the seed 2 paragraph gets rewritten.
- **Not ~8.2% at all** → a finding about the original runs' reproducibility. Report it; do not
  quietly replace the old number. This is the _expected_ outcome — see the determinism section.

## Kaggle rules (cost real GPU hours when broken)

All four Phase 3 cells are done, so no runs are planned. If one happens anyway:

- **Push `results.csv` to GitHub _before_ launching the run** — the notebook seeds its results
  file from the cloned repo, so an unpushed row is a row the next run neither skips nor carries.
- **Never put two configs in one version** — a run killed at the session limit discards
  `/kaggle/working` entirely.
- The notebook now passes `--log-dir /kaggle/working/logs`, outside the throwaway clone. Pull
  `logs/` back and commit it alongside `results.csv`, or the run's curve is lost again.
- A **rerun** of a seed already in `results.csv` is skipped by the notebook's `done()` guard.
  Reruns need that guard bypassed deliberately, not edited away by accident.
- `kernels push` **launches immediately — there is no dry-run.**
