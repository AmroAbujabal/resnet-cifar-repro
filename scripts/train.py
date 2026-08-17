"""Train a CIFAR ResNet with the He et al. 2015 schedule; log to results.csv.

Usage:
    python scripts/train.py --config configs/resnet20.yaml --seed 0
    python scripts/train.py --config configs/resnet56.yaml --seed 0 --device cuda

Every run appends one row to results.csv (the single source of truth for results)
and writes its full evaluation history to logs/<name>_seed<seed>.csv, which is
committed: a run whose curve lives only in a session's stdout is a run that
cannot be explained afterwards.
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_loaders  # noqa: E402
from src.eval import evaluate  # noqa: E402
from src.metric import top1_error  # noqa: E402
from src.model import resnet  # noqa: E402
from src.train import set_seed, train_step  # noqa: E402


def infinite(loader):
    while True:
        yield from loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--log-dir", default="logs", help="per-run evaluation logs (committed)")
    ap.add_argument("--overwrite-log", action="store_true",
                    help="replace an existing per-run log instead of refusing")
    ap.add_argument("--eval-every", type=int, default=8000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device)
    set_seed(args.seed)

    loaders = get_loaders(
        args.data_root, batch_size=cfg["batch_size"], val_size=cfg["val_split"],
        seed=args.seed, full_train=(cfg["train_split"] == 50000), num_workers=2,
        dataset=cfg.get("dataset", "cifar10"),
    )
    model = resnet(
        cfg["n"], num_classes=cfg.get("num_classes", 10), preact=cfg.get("preact", False)
    ).to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=cfg["lr_milestones_iters"], gamma=cfg["lr_gamma"],
    )

    max_iters = cfg["max_iters"]
    warmup = cfg.get("warmup", False)
    if warmup:  # ResNet-110: lr 0.01 until train err < 80%, then restore 0.1
        for g in opt.param_groups:
            g["lr"] = 0.01

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{cfg['name']}_seed{args.seed}.csv")
    # These logs are committed and some are irrecoverable, so opening "w" would
    # destroy one the instant a name+seed is re-run -- including at iteration 0 of a
    # run that then dies at the session limit, which is the case the log exists for.
    if os.path.exists(log_path) and not args.overwrite_log:
        sys.exit(f"{log_path} exists; pass --overwrite-log to replace that run's curve")
    log = open(log_path, "w", newline="")
    log_w = csv.writer(log)
    log_w.writerow(["iter", "train_error_pct", "test_error_pct"])

    print(f"{cfg['name']} | seed {args.seed} | {device} | {max_iters} iters -> {log_path}")
    start = time.time()
    it, warmed = 0, not warmup
    for x, y in infinite(loaders["train"]):
        x, y = x.to(device), y.to(device)
        loss = train_step(model, (x, y), opt)
        if not warmed:
            with torch.no_grad():
                tr_err = top1_error(model(x), y)
            if tr_err < 80.0:
                for g in opt.param_groups:
                    g["lr"] = cfg["lr"]
                warmed = True
        else:
            sched.step()
        it += 1
        if it % 1000 == 0:
            print(f"  iter {it:6d}/{max_iters}  loss {loss:.3f}  lr {opt.param_groups[0]['lr']:.4f}")
        if it % args.eval_every == 0 or it >= max_iters:
            test_err = evaluate(model, loaders["test"], device)
            # Train error on the un-augmented view, at every interval and not only
            # at the end: an elevated train curve is how an optimisation failure
            # tells itself apart from a generalisation one afterwards.
            train_err = evaluate(model, loaders["train_eval"], device)
            log_w.writerow([it, f"{train_err:.2f}", f"{test_err:.2f}"])
            log.flush()  # a session killed at its limit must not take the log with it
            print(f"  [eval] iter {it}: train error {train_err:.2f}%  "
                  f"test error {test_err:.2f}%")
        if it >= max_iters:
            break

    log.close()
    wall = time.time() - start
    print(f"DONE {cfg['name']} seed {args.seed}: test error {test_err:.2f}%, "
          f"train error {train_err:.2f}% in {wall/60:.1f} min")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": cfg["name"], "n": cfg["n"], "seed": args.seed,
        "iters": max_iters, "test_error_pct": round(test_err, 2),
        "train_error_pct": round(train_err, 2),
        "final_train_loss": round(loss, 4), "wall_min": round(wall / 60, 1),
        "device": str(device),
    }
    write_header = not os.path.exists(args.results)
    if not write_header:
        # DictWriter writes in row order regardless of what is on disk, so a results.csv
        # from an older schema would silently misalign every value in every appended row.
        with open(args.results, newline="") as f:
            header = next(csv.reader(f), None)
        assert header == list(row), f"{args.results} header {header} != {list(row)}"
    with open(args.results, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"appended row -> {args.results}")


if __name__ == "__main__":
    main()
