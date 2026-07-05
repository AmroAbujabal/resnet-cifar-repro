# ResNet CIFAR-10 Reproduction + Extension — Plan

Source: He et al. 2015, *Deep Residual Learning for Image Recognition*, arXiv:1512.03385.
PDF read in full; all facts below verified against Section 4.2 and Table 6.
**No code written yet — this is the Phase 1 extraction + Phase 2–3 task list. Awaiting confirmation.**

---

## Phase 1 — Extraction

### 1. Problem
- **Degradation problem:** as plain networks get deeper, accuracy saturates then degrades — and
  crucially the *training* error goes **up**, not just test error. So it is **not overfitting**; it is an
  optimization difficulty. A deeper net should be able to match a shallower one by setting the extra
  layers to identity (giving no-worse training error), yet solvers fail to find such a solution.
- **Residual learning:** instead of fitting the desired mapping `H(x)` directly, let the stacked layers
  fit a **residual** `F(x) := H(x) − x`; the block computes `F(x) + x` via an **identity shortcut**.
  Hypothesis: driving `F` toward 0 (to approximate identity) is easier than learning identity through
  stacked nonlinear layers. Identity shortcuts add **no parameters and no extra compute**.
- **Inputs/outputs:** input = 32×32×3 image (CIFAR); output = 10-way softmax. **Success** = lower test
  error AND demonstrating deeper residual nets get lower *training* error where deeper plain nets do not.

### 2. Datasets
- **ImageNet (ILSVRC 2012):** noted for completeness. **OUT OF SCOPE.**
- **CIFAR-10 (Sec 4.2) — reproduction target:**
  - 50k train + 10k test, 10 classes, 32×32 color.
  - **45k/5k train/val split** — the 5k val is used to *determine the training schedule*.
  - **Per-pixel mean subtraction.**
  - **Train augmentation:** 4-pixel pad each side → random 32×32 crop from the padded 40×40 image, or its
    horizontal flip (recipe from Lee et al. [24], DSN).
  - **Test:** single view of the original 32×32 image — no crop, no flip, no TTA.

### 3. Metric
- **Top-1 classification error % on the 10k test set** = 100 × (misclassified / 10,000), argmax of softmax,
  single original-image view.

| Model | n | Layers (6n+2) | Params | Reported error % | Ref |
|---|---|---|---|---|---|
| ResNet-20  | 3   | 20   | 0.27M | 8.75 | Table 6 |
| ResNet-32  | 5   | 32   | 0.46M | 7.51 | Table 6 |
| ResNet-44  | 7   | 44   | 0.66M | 7.17 | Table 6 |
| ResNet-56  | 9   | 56   | 0.85M | 6.97 | Table 6 |
| ResNet-110 | 18  | 110  | 1.7M  | 6.43 (6.61 ± 0.16, best of 5) | Table 6 |
| ResNet-1202| 200 | 1202 | 19.4M | 7.93 | Table 6 |

### 4. Model (CIFAR variant)
- Input 32×32×3, per-pixel mean subtracted.
- **First layer:** 3×3 conv, 16 filters (stride 1, pad 1) → BN → ReLU.
- **6n conv layers** in 3 stages over feature-map sizes {32, 16, 8}, filters {16, 32, 64}, **2n layers
  (= n basic blocks) per stage**:
  - Stage 1: 32×32, 16 filters, n blocks.
  - Stage 2: 16×16, 32 filters, n blocks; first block downsamples (stride-2 conv).
  - Stage 3: 8×8, 64 filters, n blocks; first block downsamples (stride-2 conv).
- **Global average pool → 64-d → 10-way FC → softmax.**
- Total weighted layers = **6n + 2**; shortcuts join pairs of 3×3 layers → **3n shortcuts**.
- **Shortcuts = Option A:** identity + zero-padding for increased dims (no extra params). On stride-2/dim
  increase: subsample identity (stride 2) and zero-pad the extra channels.
- **BN after each conv, before ReLU** (conv→BN→ReLU); add after 2nd conv's BN, then ReLU after add.
- **He/MSRA init [13]. No dropout.**
- **SGD:** momentum 0.9, weight decay 1e-4, batch size 128.
- **LR 0.1**, ÷10 at **32k** and **48k** iterations, stop at **64k** iterations.
- **ResNet-110:** warm up at **LR 0.01 until training error < 80%** (~400 iters), then back to 0.1;
  rest of schedule unchanged.

### 5. Gaps (unspecified, reproduction-relevant)
1. **Exact seeds** (init, shuffle, aug RNG). Paper's own ResNet-110 spread is ±0.16% over 5 runs.
2. **45k vs full 50k for final models.** Schedule is tuned on 45k/5k; the faithful akamaster repo trains
   final models on the **full 50k**. **Decision needed.**
3. Exact **5k val split indices**.
4. **Data ordering / shuffle** procedure per epoch.
5. **Color augmentation:** the CIFAR recipe uses **only pad+crop+flip — no color aug.** (The AlexNet-style
   color augmentation the paper defers to is in the **ImageNet** pipeline, not CIFAR — flagging because
   the task brief lists it under CIFAR gaps.)
6. **Option-A downsample mechanism:** strided subsample (`x[:, :, ::2, ::2]`) + channel zero-pad — exact
   form unspecified in paper; akamaster uses this.
7. **BN momentum/epsilon** (framework defaults).
8. **Weight decay on BN params / biases?** (often applied to all; sometimes BN excluded.)
9. **FC bias / init.**
10. **Iteration↔epoch:** 64k iters × 128 / 45k ≈ 182 epochs (or ≈164 on 50k) — depends on gap #2.

---

## Phase 2 — Reproduce CIFAR-10

**Reuse vs rebuild:**
- **Reuse (reference only):** `akamaster/pytorch_resnet_cifar10` — faithful, Option A. Mirror its
  `BasicBlock` + LambdaLayer Option-A downsample, He init, stage/block layout.
- **Rebuild ourselves (TDD):** data pipeline, metric, training loop, config, `results.csv` logging, tests.
- **Do NOT use** `torchvision.models.resnet` — that is the ImageNet stem (7×7/stride-2 + maxpool, 4 stages)
  and will not match CIFAR numbers.

**Repo structure:**
```
resnet-repro/
  data/            # CIFAR (gitignored)
  src/  model.py data.py train.py eval.py utils.py
  configs/         # resnet20/32/44/56/110.yaml
  scripts/         # run_train.sh, run_matrix.sh
  tests/           # test_data, test_metric, test_model, test_overfit
  results.csv      # single source of truth for all runs
  PLAN.md  CLAUDE.md
```

**Ordered, independently testable tasks (validate cheap first):**
- **T1 Data loader + aug** — test: shape (128,3,32,32); per-pixel mean ≈0 post-subtract; sizes 45k/5k/10k;
  train view differs from original, test view is untouched original.
- **T2 Metric** — top-1 error %; unit test on a tiny fake batch with known predictions.
- **T3 Model builder `resnet(n)`** — assert layers = 6n+2; param counts match table (0.27M @ n=3, 0.85M @
  n=9, 1.7M @ n=18); output (B,10); Option A adds 0 params.
- **T4 Single-batch overfit** — one batch of 128, loss → ~0 / train acc → 100%. Gate before any full run.
- **T5 Full training** — **ResNet-20 first** (cheap), then **ResNet-56**.

**Definition of done (tolerance stated):** ResNet-20 and ResNet-56 land within **±0.5% absolute** test
error of the paper (≤ 9.25% and ≤ 7.47% respectively), on the **mean over ≥3 seeds**.

---

## Phase 3 — Extend (only after repro validated)

**Held FIXED across all cells:** data splits, seeds, metric code, optimizer schedule, training budget, hardware.

- **A. CIFAR-100** — same 32×32×3, same 50k/10k, same pad+crop+flip; **recompute per-pixel mean**; FC head
  64→**100**. Differences: 100 classes, recomputed mean, 500 train imgs/class (harder → higher error) — but
  error% stays directly comparable (identical metric).
- **B. Pre-activation ResNet** (He et al. 2016, *Identity Mappings in Deep Residual Networks*) at **same
  depth (ResNet-56)** — only change is BN/ReLU ordering → **BN→ReLU→conv** (full pre-activation), clean
  identity shortcut. Param count identical, budget matched. Can't hold perfectly constant: placement of the
  final BN/ReLU before pooling — will be called out.
- **C. Matrix** — {original, pre-act} × {CIFAR-10, CIFAR-100} = **4 cells**. Per cell: test error% mean±std,
  train error, wall-clock. Est. ResNet-56 ≈ 1–2 GPU-hr/run → 4×3 seeds ≈ **12–24 GPU-hr**. All rows → one
  `results.csv`, no hand-copying.
- **D. Honesty** — ≥3 seeds/cell, report mean±std, flag any Δ smaller than combined std as **within noise**.

---

## Phase 4 — Confirm before any code

1. **GAPS:** (a) train final models on **full 50k** (schedule tuned on 45k/5k), akamaster-style? (b) seed set
   e.g. {0,1,2}? (c) Option-A downsample = strided subsample + channel zero-pad? (d) apply weight decay to BN
   params or exclude them? (e) confirm **no color aug** for CIFAR.
2. **CIFAR-100** as the second dataset (only head + mean change)?
3. **Pre-activation ResNet-56** as the comparison model (full pre-activation ordering)?
