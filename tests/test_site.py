"""The write-up may not drift from results.csv.

The page claims every number on it is read from the results file. That claim is
only true while scripts/build_site.py has been run since the last append, so the
check runs here rather than depending on anyone remembering.
"""
import subprocess
import sys


def test_site_matches_results_csv():
    r = subprocess.run([sys.executable, "scripts/build_site.py", "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
