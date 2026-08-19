"""Regenerate the results-derived numbers in site/index.html from results.csv.

The write-up claims that no *result* on it was transcribed by hand. This script is
what makes that true: it recomputes the statistics from `results.csv` (and the
per-run evaluation logs in `logs/` for Figure 3) and writes them back into

  * the `<span data-stat="KEY">` placeholders in the prose and tables, and
  * the figure data block between the GENERATED markers in the page's script.

Run it after any run appends to results.csv:

    python scripts/build_site.py            # rewrite the page
    python scripts/build_site.py --check    # fail if the page is out of date

Not everything numeric on the page comes from here, and the distinction matters:
paper values (8.75%, 6.97%) are quoted from Table 6 of He et al. 2015, parameter
counts are asserted by the test suite, and the figure axis bounds are chosen. What
this script owns is every number that is a *result of a run*.
"""
import argparse
import csv
import math
import os
import re
import statistics as st
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results.csv")
LOGS = os.path.join(ROOT, "logs")
PAGE = os.path.join(ROOT, "site", "index.html")

PAPER = {"resnet20": 8.75, "resnet56": 6.97}  # He et al. 2015, Table 6
TOLERANCE = 0.5  # pre-registered, fixed before training
FIG3 = "preact56_c100"  # the one Phase 3 configuration whose eval log survives
FIG4 = "resnet56_rerun_seed2"  # the only run with a train curve as well as a test one
MINUS = "−"  # typographic minus, matches the page's &minus;


def signed(v, d=2):
    return f"{MINUS if v < 0 else '+'}{abs(v):.{d}f}"


def load_runs():
    with open(RESULTS, newline="") as f:
        rows = list(csv.DictReader(f))
    runs = {}
    for r in rows:
        key = (r["model"], int(r["seed"]))
        # A duplicate (model, seed) means a rerun was appended. Which row the
        # published mean should use is a judgement call, not a default, so stop
        # rather than silently pick one. sys.exit, not assert: python -O would
        # strip an assert and turn a documented refusal into last-row-wins.
        if key in runs:
            sys.exit(f"{RESULTS} has two rows for {key}. A rerun needs an explicit rule "
                     "for which row the reported aggregate uses -- see CLAUDE.md.")
        runs[key] = r
    return rows, runs


def cell(runs, model):
    """Per-model summary, seeds in ascending order."""
    seeds = sorted(s for m, s in runs if m == model)
    test = [float(runs[(model, s)]["test_error_pct"]) for s in seeds]
    train = [runs[(model, s)]["train_error_pct"] for s in seeds]
    train = [float(t) for t in train if t != ""]
    wall = [float(runs[(model, s)]["wall_min"]) for s in seeds]
    # A one-run cell has no spread. resnet56_rerun is exactly that, so this is not
    # a hypothetical: st.stdev would raise and take the whole page build with it.
    sd = st.stdev(test) if len(test) > 1 else None
    return {
        "seeds": seeds,
        "test": test,
        "train": train,
        "mean": st.mean(test),
        "sd": sd,
        "se": sd / math.sqrt(len(test)) if sd is not None else None,
        "wall_h": st.mean(wall) / 60.0,
    }


def stats(rows, runs):
    models = sorted({m for m, _ in runs})
    c = {m: cell(runs, m) for m in models}
    out = {}

    for m, d in c.items():
        out[f"{m}.mean"] = f"{d['mean']:.2f}"
        if d["sd"] is None:  # a one-run cell has no spread; publish no spread keys
            out[f"{m}.meansd"] = f"{d['mean']:.2f}%"
        else:
            out[f"{m}.sd"] = f"{d['sd']:.2f}"
            out[f"{m}.se"] = f"{d['se']:.2f}"
            out[f"{m}.meansd"] = f"{d['mean']:.2f} ± {d['sd']:.2f}%"
        out[f"{m}.seeds"] = " / ".join(f"{v:.2f}" for v in d["test"])
        for i, v in zip(d["seeds"], d["test"]):
            out[f"{m}.seed{i}"] = f"{v:.2f}"
        # Empty train error is "never measured", not zero -- say so.
        if not d["train"]:
            out[f"{m}.train"] = "not measured"
        elif min(d["train"]) == max(d["train"]):
            out[f"{m}.train"] = f"{d['train'][0]:.2f}%"
        else:
            out[f"{m}.train"] = f"{min(d['train']):.2f} to {max(d['train']):.2f}%"
        if m in PAPER:
            out[f"{m}.delta"] = signed(d["mean"] - PAPER[m])

    r56 = c["resnet56"]
    margin = TOLERANCE - abs(r56["mean"] - PAPER["resnet56"])
    out["resnet56.margin"] = f"{margin:.2f}"
    out["resnet56.margin_se"] = f"{margin / r56['se']:.2f}"
    out["resnet56.seed_gap"] = f"{max(r56['test']) - min(r56['test']):.2f}"
    # Seeds that would have bought a standard error of 0.1 points, and what
    # that would have cost at this run's measured wall clock.
    n = math.ceil((r56["sd"] / 0.1) ** 2)
    out["resnet56.n_for_se01"] = str(n)
    out["resnet56.hours_for_se01"] = f"{round(n * r56['wall_h'] / 10) * 10:.0f}"

    # Phase 4: the same seed label, rerun deterministically. The gap between the two
    # is the whole finding, so it is computed rather than written into the prose.
    if "resnet56_rerun" in c:
        seed = c["resnet56_rerun"]["seeds"][0]
        original = c["resnet56"]["test"][c["resnet56"]["seeds"].index(seed)]
        out["resnet56.rerun_gap"] = f"{abs(original - c['resnet56_rerun']['test'][0]):.2f}"

    for tag, orig, pre in (("cifar10", "resnet56", "preact56"),
                           ("cifar100", "resnet56_c100", "preact56_c100")):
        delta = c[pre]["mean"] - c[orig]["mean"]
        comb = math.hypot(c[orig]["sd"], c[pre]["sd"])
        out[f"delta.{tag}"] = signed(delta)
        out[f"absdelta.{tag}"] = f"{abs(delta):.2f}"
        out[f"combsd.{tag}"] = f"{comb:.2f}"
        out[f"ratio.{tag}"] = f"{abs(delta) / comb:.2f}"

    c100 = c["resnet56_c100"]["train"] + c["preact56_c100"]["train"]
    out["c100.train_lo"] = f"{min(c100):.2f}"
    out["c100.train_hi"] = f"{max(c100):.2f}"

    # The page prints these next to numbers derived from them, so they are served
    # from the same constants rather than typed twice.
    for m, v in PAPER.items():
        out[f"paper.{m}"] = f"{v:.2f}"
    out["tolerance"] = f"{TOLERANCE:.1f}"

    # The Section 5 sentence about the first decay quotes four points off the
    # rerun's curve. They come from the log, not from reading the figure.
    curve = {i: (tr, te) for i, tr, te in rerun_curve()}
    for it in (32000, 40000):
        out[f"rerun.train_at_{it // 1000}k"] = f"{curve[it][0]:.2f}"
        out[f"rerun.test_at_{it // 1000}k"] = f"{curve[it][1]:.2f}"

    out["runs.total"] = str(len(rows))
    # The Limitations bullet about retained logs states this as a fraction. Both
    # halves have to be computed, or the appended row moves one and not the other.
    out["runs.logged"] = str(len([f for f in os.listdir(LOGS) if "_seed" in f]))
    out["runs.gpu_hours"] = f"{sum(float(r['wall_min']) for r in rows) / 60:.0f}"
    return out, c


def curves():
    """Figure 3: test error against iteration, per seed, from the retained logs."""
    series = []
    for name in sorted(f for f in os.listdir(LOGS) if f.startswith(FIG3 + "_seed")):
        with open(os.path.join(LOGS, name), newline="") as f:
            pts = [(int(r["iter"]), float(r["test_error_pct"])) for r in csv.DictReader(f)]
        series.append(pts)
    if not series:
        sys.exit(f"no {FIG3} logs in {LOGS} -- Figure 3 would render blank")
    return series


def rerun_curve():
    """Figure 4: train and test error against iteration, from the Phase 4 rerun."""
    path = os.path.join(LOGS, FIG4 + ".csv")
    if not os.path.exists(path):
        sys.exit(f"{path} is missing -- Figure 4 would render blank")
    with open(path, newline="") as f:
        return [(int(r["iter"]), float(r["train_error_pct"]), float(r["test_error_pct"]))
                for r in csv.DictReader(f)]


def js_block(c):
    lines = ["        // <<< GENERATED from results.csv + logs/ by scripts/build_site.py.",
             "        // Do not edit by hand: rerun the script after any new run appends.",
             "        var REPRO = ["]
    for m, label in (("resnet20", "ResNet-20"), ("resnet56", "ResNet-56")):
        lines.append("          { label: %r, paper: %s, seeds: [%s] }," % (
            label, PAPER[m], ", ".join(str(v) for v in c[m]["test"])))
    lines.append("        ];")
    lines.append("")
    lines.append("        var MATRIX = [")
    for panel, lo, hi, orig, pre in (
        ("CIFAR-10", 6.4, 8.6, "resnet56", "preact56"),
        # Same 2.2 point span in both panels, so equal bar width means equal spread.
        ("CIFAR-100", 28.9, 31.1, "resnet56_c100", "preact56_c100"),
    ):
        lines.append("          {")
        lines.append("            panel: %r," % panel)
        lines.append(f"            lo: {lo},")
        lines.append(f"            hi: {hi},")
        lines.append("            step: 0.5,")
        lines.append("            rows: [")
        for label, var, model in (("original", "--sa", orig), ("pre-act", "--sb", pre)):
            lines.append("              { label: %r, color: 'var(%s)', seeds: [%s] }," % (
                label, var, ", ".join(str(v) for v in c[model]["test"])))
        lines.append("            ],")
        lines.append("          },")
    lines.append("        ];")
    lines.append("")
    lines.append("        var CURVES = [")
    for pts in curves():
        lines.append("          [%s]," % ", ".join(f"[{i}, {e}]" for i, e in pts))
    lines.append("        ];")
    lines.append("")
    lines.append("        var RERUN = [%s];" % ", ".join(
        f"[{i}, {tr}, {te}]" for i, tr, te in rerun_curve()))
    lines.append("        // >>> END GENERATED")
    return "\n".join(lines).replace("'", '"')


def skeleton(html):
    """The page minus the two regions this script owns: span bodies and the figure
    block. Everything else has to come through a build byte-identical, and that --
    not counting tags -- is the invariant. It catches a deleted row, a deleted
    figure, eaten prose, a mangled attribute: anything outside what we meant to
    touch. It fails safe, because under-stripping raises a false alarm rather than
    passing silently."""
    html = re.sub(r'(data-stat="[^"]+"[^>]*>)[^<]*', r"\1", html)
    return re.sub(r"// <<< GENERATED.*?// >>> END GENERATED", "", html, flags=re.S)


def render(html, values, c):
    original = html
    present = Counter(re.findall(r'data-stat="([^"]+)"', html))
    unknown = set(present) - set(values)
    if unknown:
        sys.exit(f"page asks for stats this script does not compute: {sorted(unknown)}")

    for key, expected in present.items():
        # The body is [^<]*, never .*? -- a data-stat span holds plain text, and a
        # formatter may close it as "</span\n>". With .*? and a literal "</span>",
        # the non-greedy body runs past such a close to the next literal one and the
        # substitution eats every tag in between. That silently deleted a table row,
        # a whole figure and the opening of a paragraph before this was caught.
        pattern = re.compile(r'(<span[^>]*data-stat="%s"[^>]*>)([^<]*)(</span\s*>)'
                             % re.escape(key), re.S)
        html, n = pattern.subn(lambda m: m.group(1) + values[key] + m.group(3), html)
        if n != expected:
            sys.exit(f"{key}: substituted {n} of {expected} spans -- the page's markup "
                     "is not what this script expects; fix it before publishing")

    html = re.sub(r"        // <<< GENERATED.*?// >>> END GENERATED",
                  lambda _: js_block(c), html, flags=re.S)
    if skeleton(original) != skeleton(html):
        sys.exit("this build changed the page outside the spans and the figure block. "
                 "That is how a table row and a whole figure were once deleted; "
                 "nothing is written.")
    check_structure(html)
    return html


def check_structure(html):
    """A standing audit of the page as it stands, which skeleton() cannot give:
    skeleton() compares a build against its own input, so it is blind to damage that
    arrived by some other route and was committed. This is the only thing that looks
    at the page on its own terms."""
    for tag in ("table", "tbody", "tr", "td", "figure", "div", "p", "ul", "li", "span"):
        opened = len(re.findall(r"<%s[\s>]" % tag, html))
        closed = len(re.findall(r"</%s\s*>" % tag, html))
        if opened != closed:
            sys.exit(f"<{tag}> is unbalanced: {opened} open, {closed} close")
    for anchor in ('id="fig1"', 'id="fig2"', 'id="fig3"', 'id="fig4"'):
        if anchor not in html:
            sys.exit(f"{anchor} is missing -- a figure would render blank and silently")


def page_values(html):
    """What the page currently states: displayed statistics, and the numbers in the
    figure block. Compared instead of the raw bytes because the page is run through
    a formatter, and a reflowed line is not a stale number."""
    # A list, not a dict: several keys appear more than once on the page, and a dict
    # keeps only the last, so a hand-edited number in the abstract would compare equal
    # to the same key's correct value in a table and --check would report "current".
    spans = [(k, re.sub(r"\s+", " ", v).strip()) for k, v in
             re.findall(r'data-stat="([^"]+)"[^>]*>([^<]*)</span\s*>', html, re.S)]
    block = re.search(r"// <<< GENERATED.*?// >>> END GENERATED", html, re.S)
    if not block:
        sys.exit("page has lost its GENERATED markers")
    return spans, re.findall(r"-?\d+\.?\d*", block.group(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the page is not what this script would write")
    args = ap.parse_args()

    rows, runs = load_runs()
    values, c = stats(rows, runs)
    with open(PAGE, encoding="utf-8") as f:
        html = f.read()
    built = render(html, values, c)

    if page_values(html) == page_values(built):
        print(f"site/index.html is current with {len(rows)} runs in results.csv")
        return
    if args.check:
        sys.exit(f"{PAGE} is stale -- run scripts/build_site.py")
    with open(PAGE, "w", encoding="utf-8") as f:
        f.write(built)
    print(f"site/index.html rebuilt from {len(rows)} runs in results.csv")


if __name__ == "__main__":
    main()
