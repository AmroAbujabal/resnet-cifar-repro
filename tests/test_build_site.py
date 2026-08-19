"""The two guards that stop the page from lying must actually fire.

Both were written as refusals -- a duplicate (model, seed) and a data-stat key the
script cannot compute -- and neither was exercised by anything, which is a poor
state for the only thing standing between results.csv and a published number.
"""
import os

import pytest

import build_site as bs  # scripts/ is on sys.path via conftest.py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_duplicate_model_seed_stops_the_build(tmp_path, monkeypatch):
    rows = open(os.path.join(ROOT, "results.csv")).read().splitlines(keepends=True)
    dup = tmp_path / "results.csv"
    dup.write_text("".join(rows + [rows[1]]))  # same (model, seed) twice
    monkeypatch.setattr(bs, "RESULTS", str(dup))
    with pytest.raises(SystemExit):
        bs.load_runs()


def test_unknown_stat_key_stops_the_build():
    with pytest.raises(SystemExit):
        bs.render('<span data-stat="not.a.real.stat">0.00</span>', {}, {})


def test_deleted_markup_stops_the_build():
    # The failure that shipped: a span rewrite ate a table row and a figure, and
    # every value still on the page compared equal, so --check said "current".
    html = open(os.path.join(ROOT, "site", "index.html")).read()
    with pytest.raises(SystemExit):
        bs.check_structure(html.replace("</table>", "", 1))
    with pytest.raises(SystemExit):
        bs.check_structure(html.replace('id="fig2"', 'id="gone"'))
