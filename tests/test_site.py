"""The write-up may not drift from results.csv.

The page claims every number on it is read from the results file. That claim is
only true while scripts/build_site.py has been run since the last append, so the
check runs here rather than depending on anyone remembering.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_site_matches_results_csv():
    # Absolute path: a cwd-relative one fails from anywhere but the repo root, and
    # reports it as a stale page.
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_site.py"),
                        "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
