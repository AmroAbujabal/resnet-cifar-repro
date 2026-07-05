# ResNet CIFAR-10 Reproduction — repo guide

Reproduce He et al. 2015 (arXiv:1512.03385) CIFAR-10 Table 6, then extend to CIFAR-100
and pre-activation ResNet-56. Full spec + task list in `PLAN.md`.

## Environment
- **This machine (Intel i5, 2 cores, no GPU) is for code + tests only.** T1–T4 run on CPU in seconds.
- **Full training (T5) runs on cloud GPU** (Colab T4). CPU training here is infeasible (~a day/run).
- Interpreter: `/usr/local/bin/python3.11`. Use a project venv: `.venv/` (gitignored).
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

## Definition of done (Phase 2)
ResNet-20 and ResNet-56 within **±0.5% absolute** of paper error (≤9.25%, ≤7.47%),
mean over ≥3 seeds.

## Run tests
`.venv/bin/python -m pytest tests/ -q`
