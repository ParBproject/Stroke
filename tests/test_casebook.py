from casebook.analyze import build


def test_prevalence_and_split():
    payload = build()
    data = payload["dataset"]
    assert 0.04 < data["prevalence"] < 0.06
    assert data["strokes"] == 249
    assert abs(data["test_rows"] / data["rows"] - 0.2) < 0.02
    assert data["bmi_missing"] > 100


def test_baselines_and_desk_calibration():
    payload = build()
    models = {row["name"]: row for row in payload["models"]}
    assert models["Majority rate"]["at_half"]["sensitivity"] == 0
    assert models["Age only"]["roc_auc"] > 0.75
    assert models["Logistic"]["roc_auc"] >= models["Age only"]["roc_auc"] - 0.02
    assert models["Logistic"]["brier"] < 0.08
    assert models["Logistic, class-weighted"]["brier"] > 0.12
    top = payload["calibration"][0]
    assert top["mean_p"] < 0.08
    assert top["observed"] < 0.05


def test_age_odds_and_operating_point():
    payload = build()
    age = next(row for row in payload["odds_ratios"] if row["term"].startswith("Age"))
    assert age["odds_ratio"] > 1.5
    assert payload["operating"]["sensitivity"] >= 0.70
    assert payload["desk_model"] == "Logistic"
    terms = [row["term"] for row in payload["odds_ratios"]]
    assert any("Glucose" in term or "BMI" in term for term in terms)
