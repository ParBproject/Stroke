"""Holdout casebook checks. The fit is deterministic at seed 42."""

import json

import pytest

from casebook.analyze import OUT, choose_operating


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
    age = models["Age only"]
    desk = models["Logistic"]
    assert age["roc_auc"] > 0.75
    assert desk["roc_auc"] >= age["roc_auc"] - 0.02
    assert payload["desk_model"] == "Logistic"
    assert age["scores_are_risks"] is True
    assert desk["scores_are_risks"] is True
    assert models["Logistic, class-weighted"]["scores_are_risks"] is False
    # Class-weighted age scores used to post a Brier near 0.17. That was not a probability.
    assert age["brier"] < 0.06
    assert abs(age["brier"] - desk["brier"]) < 0.01
    assert age["at_half"]["sensitivity"] == 0
    assert payload["desk_fit"]["penalty"] == "l2"
    assert payload["desk_fit"]["C"] == 1.0
    assert payload["desk_fit"]["class_weight"] is None


def test_holdout_intervals_match_the_counts(payload):
    from casebook.analyze import wilson_interval

    operating = payload["operating"]
    assert operating["sensitivity_ci"] == wilson_interval(
        operating["tp"], operating["tp"] + operating["fn"]
    )
    assert operating["specificity_ci"] == wilson_interval(
        operating["tn"], operating["tn"] + operating["fp"]
    )
    assert operating["ppv_ci"] == wilson_interval(operating["tp"], operating["tp"] + operating["fp"])
    assert operating["sensitivity_ci"][1] - operating["sensitivity_ci"][0] > 0.1
    row = next(item for item in payload["thresholds"] if item["threshold"] == operating["threshold"])
    assert row["sensitivity_ci"] == operating["sensitivity_ci"]


def test_age_lift_is_interval_not_a_point(payload):
    lift = payload["vs_age"]
    assert "not refit" in lift["method"]
    assert lift["resamples"] == 2000
    assert lift["seed"] == 42
    assert lift["roc_auc_ci"][0] <= lift["roc_auc"] <= lift["roc_auc_ci"][1]
    assert lift["pr_auc_ci"][0] <= lift["pr_auc"] <= lift["pr_auc_ci"][1]
    assert lift["brier_ci"][0] <= lift["brier"] <= lift["brier_ci"][1]
    # Fifty holdout strokes do not separate the ROC curves. PR-AUC does move.
    assert lift["roc_auc_ci"][0] <= 0 <= lift["roc_auc_ci"][1]
    assert lift["pr_auc_ci"][0] > 0
    models = _models(payload)
    for name in ("Age only", "Logistic"):
        model = models[name]
        assert model["roc_auc_ci"][0] <= model["roc_auc"] <= model["roc_auc_ci"][1]
        assert model["roc_auc_ci"][1] - model["roc_auc_ci"][0] > 0.05


def test_calibration_resolves_the_operating_region(payload):
    rows = payload["calibration"]
    n = payload["dataset"]["test_rows"]
    assert max(row["n"] for row in rows) < 0.5 * n
    threshold = payload["operating"]["threshold"]
    assert any(row["hi"] < threshold for row in rows)
    assert any(row["lo"] <= threshold <= row["hi"] for row in rows)
    for row in rows:
        assert row["observed_ci"][0] <= row["observed"] <= row["observed_ci"][1]
        assert "events" in row
    summary = payload["calibration_summary"]
    assert summary["scheme"] == "equal_count"
    assert summary["adjusts_probabilities"] is False
    assert summary["fit_on"] == "holdout"
    assert summary["slope_ci"][0] <= summary["slope"] <= summary["slope_ci"][1]
    assert summary["slope_ci"][0] < 1 < summary["slope_ci"][1]


def test_odds_ratios_are_not_the_desk_model(payload):
    meta = payload["association_model"]
    assert meta["is_desk_model"] is False
    assert meta["penalized"] is False
    separation = meta["separation"]
    assert separation["level"] == "Never_worked"
    assert separation["train_strokes"] == 0
    assert separation["train_rows"] > 0
    assert "Never_worked" in payload["odds_design"]
    assert "gender_Other" not in payload["odds_design"]
    assert "those levels separated" not in payload["odds_design"]
    terms = " ".join(row["term"] for row in payload["odds_ratios"])
    assert "Work" not in terms


def test_age_odds_ratio(payload):
    age = next(row for row in payload["odds_ratios"] if row["term"].startswith("Age"))
    assert age["odds_ratio"] > 1
    assert age["ci_low"] > 1
    assert age["ci_high"] < 20
    terms = " ".join(row["term"] for row in payload["odds_ratios"])
    assert "work_type" not in terms
    assert "Other" not in terms


def test_operating_point_is_frozen_on_training_oof(payload):
    """The published cutoff is the training rule, scored once on the holdout."""
    grid = payload["thresholds"]
    assert any(abs(row["threshold"] - 0.15) < 1e-9 for row in grid)
    selection = payload["threshold_selection"]
    assert selection["source"] == "train_oof"
    assert selection["folds"] == 5
    assert selection["seed"] == 42
    assert selection["labeled_rows"] == payload["dataset"]["train_rows"]
    assert selection["labeled_rows"] != payload["dataset"]["test_rows"]
    if selection["oof_sensitivity"] >= 0.70:
        assert "sensitivity >= 0.70" in selection["rule"]
    operating = payload["operating"]
    assert operating["threshold"] == selection["threshold"]
    assert operating["selected_on"] == "train_oof"
    assert "holdout" in operating["rule"]
    assert "training" in operating["selection_note"]
    holdout_row = next(
        row for row in grid if abs(row["threshold"] - operating["threshold"]) < 1e-9
    )
    assert operating["sensitivity"] == holdout_row["sensitivity"]
    assert operating["tp"] == holdout_row["tp"]
    assert operating["flagged"] == holdout_row["flagged"]
    assert "per_1000" in operating
    # The slider grid is not the selection set. Do not require holdout recall >= 0.70.


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
