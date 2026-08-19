"""The write-up may not drift from results.csv.

The page claims every number on it is read from the results file. That claim is
only true while scripts/build_site.py has been run since the last append, so the
check runs here rather than depending on anyone remembering.
"""
import subprocess
import sys

import build_site as bs  # scripts/ is on sys.path via conftest.py


def test_site_matches_results_csv():
    # bs.__file__, not a cwd-relative path: the latter fails from anywhere but the
    # repo root and reports it as a stale page.
    r = subprocess.run([sys.executable, bs.__file__, "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
