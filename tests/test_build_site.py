"""The two guards that stop the page from lying must actually fire.

Both were written as refusals -- a duplicate (model, seed) and a data-stat key the
script cannot compute -- and neither was exercised by anything, which is a poor
state for the only thing standing between results.csv and a published number.
"""
import re

import pytest

import build_site as bs  # scripts/ is on sys.path via conftest.py


def test_duplicate_model_seed_stops_the_build(tmp_path, monkeypatch):
    rows = open(bs.RESULTS).read().splitlines(keepends=True)
    dup = tmp_path / "results.csv"
    dup.write_text("".join(rows + [rows[1]]))  # same (model, seed) twice
    monkeypatch.setattr(bs, "RESULTS", str(dup))
    with pytest.raises(SystemExit):
        bs.load_runs()


def test_unknown_stat_key_stops_the_build():
    with pytest.raises(SystemExit):
        bs.render('<span data-stat="not.a.real.stat">0.00</span>', {}, {})


def test_deleted_markup_is_visible_to_skeleton():
    # The failure that shipped: a span rewrite ate a table row and a figure, and
    # every value still on the page compared equal, so --check said "current".
    # skeleton() is what sees it -- tag counting only caught that one by accident,
    # since a swallow between two adjacent spans eats a <span and a </span> both.
    html = open(bs.PAGE).read()
    row = re.search(r"<tr>\s*<td><i class=\"swatch sw-b\"></i>Pre-activation</td>.*?</tr>",
                    html, re.S).group(0)
    assert bs.skeleton(html) != bs.skeleton(html.replace(row, "", 1))
    figure = re.search(r"<figure class=\"breakout\">.*?</figure>", html, re.S).group(0)
    assert bs.skeleton(html) != bs.skeleton(html.replace(figure, "", 1))
    # ...and it must not fire on the thing the script legitimately changes.
    assert bs.skeleton(html) == bs.skeleton(html.replace("7.45 ± 0.69%", "9.99 ± 9.99%"))


def test_structure_audit_catches_a_committed_page_that_is_already_broken():
    html = open(bs.PAGE).read()
    with pytest.raises(SystemExit):
        bs.check_structure(html.replace("</table>", "", 1))
    with pytest.raises(SystemExit):
        bs.check_structure(html.replace('id="fig2"', 'id="gone"'))
