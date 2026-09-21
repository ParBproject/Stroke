"""Holdout casebook checks. The fit is deterministic at seed 42."""

import json

import pytest

from casebook.analyze import OUT, choose_operating, export


@pytest.fixture(scope="module")
def payload():
    return export()


def _models(payload):
    return {model["name"]: model for model in payload["models"]}


def test_prevalence(payload):
    assert 0.04 < payload["dataset"]["prevalence"] < 0.06
    assert payload["dataset"]["strokes"] == 249
    assert payload["dataset"]["rows"] == 5110


def test_holdout_is_about_twenty_percent(payload):
    data = payload["dataset"]
    fraction = data["test_rows"] / data["rows"]
    assert abs(fraction - 0.2) < 0.015
    assert data["test_strokes"] > 0


def test_bmi_imputed_from_train_not_full_file(payload):
    imputation = payload["dataset"]["imputation"]
    assert imputation["fit_on"] == "train"
    assert imputation["strategy"] == "median"
    assert imputation["pipeline_statistic"] == pytest.approx(imputation["train_median"])
    assert abs(imputation["train_median"] - imputation["full_median"]) > 1e-6


def test_majority_recall_at_half_is_zero(payload):
    majority = _models(payload)["Majority rate"]
    assert majority["at_half"]["sensitivity"] == 0


def test_age_and_unweighted_logistic(payload):
    models = _models(payload)
    age = models["Age only"]["roc_auc"]
    desk = models["Logistic"]["roc_auc"]
    assert age > 0.75
    assert desk >= age - 0.02
    assert payload["desk_model"] == "Logistic"
    assert models["Logistic"]["scores_are_risks"] is True
    assert models["Logistic, class-weighted"]["scores_are_risks"] is False


def test_desk_calibration_is_not_the_weighted_pattern(payload):
    top = payload["calibration"][-1]
    assert not (top["mean_p"] > 0.8 and top["observed"] < 0.3)


def test_age_odds_ratio(payload):
    age = next(row for row in payload["odds_ratios"] if row["term"].startswith("Age"))
    assert age["odds_ratio"] > 1
    assert age["ci_low"] > 1
    assert age["ci_high"] < 20
    terms = " ".join(row["term"] for row in payload["odds_ratios"])
    assert "work_type" not in terms
    assert "Other" not in terms


def test_operating_point_meets_sensitivity_when_possible(payload):
    grid = payload["thresholds"]
    assert any(abs(row["threshold"] - 0.15) < 1e-9 for row in grid)
    if any(row["sensitivity"] >= 0.70 for row in grid):
        assert payload["operating"]["sensitivity"] >= 0.70
    expected = choose_operating(grid)
    assert payload["operating"]["threshold"] == expected["threshold"]
    assert payload["operating"]["flagged"] == expected["flagged"]
    assert "per_1000" in payload["operating"]


def test_choose_operating_keeps_lowest_flag():
    grid = [
        {"threshold": 0.10, "sensitivity": 0.80, "flagged": 50, "f1": 0.2},
        {"threshold": 0.20, "sensitivity": 0.70, "flagged": 30, "f1": 0.3},
        {"threshold": 0.30, "sensitivity": 0.60, "flagged": 10, "f1": 0.9},
    ]
    picked = choose_operating(grid)
    assert picked["threshold"] == 0.20


def test_metrics_json_written(payload):
    assert OUT.exists()
    raw = OUT.read_text()
    assert "Infinity" not in raw
    assert "NaN" not in raw
    saved = json.loads(raw)
    assert saved["dataset"]["prevalence"] == payload["dataset"]["prevalence"]
    assert saved["operating"]["threshold"] == payload["operating"]["threshold"]
