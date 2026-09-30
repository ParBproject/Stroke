"""Fast checks for intervals, the age baseline, and the odds-ratio claim."""

import numpy as np
import pytest

from casebook.analyze import (
    ROOT,
    calibration,
    make_age_only,
    make_logistic,
    wilson_interval,
)


def test_wilson_interval_for_forty_of_fifty():
    interval = wilson_interval(40, 50)
    assert interval[0] == pytest.approx(0.6696, abs=0.0002)
    assert interval[1] == pytest.approx(0.8876, abs=0.0002)
    assert wilson_interval(0, 0) is None


def test_equal_count_bins_do_not_swallow_the_low_scores():
    rng = np.random.default_rng(0)
    proba = np.concatenate([rng.uniform(0.0, 0.08, 900), np.array([0.9, 0.92, 0.95])])
    labels = np.zeros(len(proba), dtype=int)
    labels[:10] = 1
    rows = calibration(labels, proba, bins=5)
    assert max(row["n"] for row in rows) < 0.5 * len(proba)
    assert sum(row["n"] for row in rows) == len(proba)
    assert rows[0]["hi"] < 0.2


def test_age_baseline_is_an_unweighted_probability():
    age = make_age_only()
    assert age.class_weight is None
    assert age.C == 1.0
    desk = make_logistic()
    clf = desk.named_steps["clf"]
    assert clf.class_weight is None
    assert clf.C == 1.0
    assert clf.l1_ratio == 0.0
    weighted = make_logistic(class_weight="balanced")
    assert weighted.named_steps["clf"].class_weight == "balanced"


def test_page_does_not_repeat_the_separation_claim():
    script = (ROOT / "casebook" / "casebook.js").read_text()
    readme = (ROOT / "README.md").read_text()
    assert "those levels separated" not in script
    assert "gender Other were omitted" not in script
    assert "class-weighted, for ranking" not in readme
    assert "Never_worked" in readme
    assert "Wilson" in readme
