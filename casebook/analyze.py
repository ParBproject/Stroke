"""Holdout stroke-risk casebook.

Train-only imputation, age-only and majority baselines, logistic odds ratios,
tree and forest comparison, threshold sweep, calibration. Not a clinical tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "healthcare-dataset-stroke-data.csv"
OUT = ROOT / "casebook" / "metrics.json"
SEED = 42
NUM = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CAT = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    frame["bmi"] = pd.to_numeric(frame["bmi"], errors="coerce")
    frame["stroke"] = frame["stroke"].astype(int)
    return frame.drop(columns=["id"])


def split(frame: pd.DataFrame):
    return train_test_split(
        frame.drop(columns=["stroke"]),
        frame["stroke"],
        test_size=0.2,
        random_state=SEED,
        stratify=frame["stroke"],
    )


def _encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(*, scale: bool) -> ColumnTransformer:
    numeric: list = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric), NUM),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("oh", _encoder()),
                    ]
                ),
                CAT,
            ),
        ]
    )


def feature_names(pre: ColumnTransformer) -> list[str]:
    names = list(NUM)
    encoder = pre.named_transformers_["cat"].named_steps["oh"]
    names.extend(encoder.get_feature_names_out(CAT).tolist())
    return names


def json_float(value, digits: int = 4):
    if value is None:
        return None
    number = float(value)
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, digits)


def json_p(value):
    """Keep tiny p-values from collapsing to 0.0."""
    if value is None:
        return None
    number = float(value)
    if np.isnan(number) or np.isinf(number):
        return None
    return float(f"{number:.6g}")


def downsample_curve(fpr, tpr, limit: int = 48) -> list[dict]:
    idx = np.linspace(0, len(fpr) - 1, min(limit, len(fpr))).astype(int)
    return [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in idx
    ]


def rates_by(frame: pd.DataFrame, column: str) -> list[dict]:
    grouped = (
        frame.groupby(column, dropna=False)["stroke"]
        .agg(n="size", strokes="sum")
        .reset_index()
    )
    grouped["rate"] = grouped["strokes"] / grouped["n"]
    rows = []
    for _, row in grouped.sort_values("rate").iterrows():
        rows.append(
            {
                "slice": str(row[column]),
                "n": int(row["n"]),
                "strokes": int(row["strokes"]),
                "rate": round(float(row["rate"]), 4),
            }
        )
    return rows


def age_bands(frame: pd.DataFrame) -> list[dict]:
    bins = [0, 18, 40, 50, 60, 70, 80, 120]
    labels = ["0–17", "18–39", "40–49", "50–59", "60–69", "70–79", "80+"]
    band = pd.cut(frame["age"], bins=bins, labels=labels, right=False)
    tmp = frame.assign(age_band=band)
    grouped = (
        tmp.groupby("age_band", dropna=False, observed=False)["stroke"]
        .agg(n="size", strokes="sum")
        .reindex(labels)
    )
    rows = []
    for label, row in grouped.iterrows():
        n = int(row["n"])
        strokes = int(row["strokes"])
        rows.append(
            {
                "slice": str(label),
                "n": n,
                "strokes": strokes,
                "rate": round((strokes / n) if n else 0.0, 4),
            }
        )
    return rows


def classification_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else 0.0
    flagged = int(tp + fp)
    return {
        "threshold": round(threshold, 2),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "ppv": round(float(ppv), 4),
        "npv": round(float(npv), 4),
        "f1": round(float(f1), 4),
        "flagged": flagged,
        "flag_rate": round(flagged / len(y_true), 4),
        "false_flags_per_true": json_float(fp / tp, 2) if tp else None,
    }


def threshold_grid(y_true, proba) -> list[dict]:
    """Risk thresholds on the unweighted logistic scale.

    Predicted risks on this file mostly sit below 0.50, so the grid stops there.
    Step 0.01 keeps a query such as ?t=0.15 on a real grid point.
    """
    rows = []
    for threshold in np.round(np.arange(0.02, 0.501, 0.01), 2):
        rows.append(classification_at(y_true, proba, float(threshold)))
    return rows


def calibration(y_true, proba, bins: int = 8) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba >= lo) & (proba < hi if hi < 1 else proba <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "lo": round(float(lo), 2),
                "hi": round(float(hi), 2),
                "n": int(mask.sum()),
                "mean_p": round(float(proba[mask].mean()), 4),
                "observed": round(float(y_true[mask].mean()), 4),
            }
        )
    return rows


def odds_ratios(train_x: pd.DataFrame, train_y: pd.Series) -> list[dict]:
    """Unweighted associations. Rare levels that separate are left out."""
    work = train_x.copy()
    work["bmi"] = work["bmi"].fillna(work["bmi"].median())
    work = work[work["gender"] != "Other"]
    y = train_y.loc[work.index]
    design = pd.DataFrame(
        {
            "age_per_10y": work["age"] / 10.0,
            "glucose_per_10": work["avg_glucose_level"] / 10.0,
            "bmi_per_5": work["bmi"] / 5.0,
            "hypertension": work["hypertension"].astype(float),
            "heart_disease": work["heart_disease"].astype(float),
            "married": (work["ever_married"] == "Yes").astype(float),
            "urban": (work["Residence_type"] == "Urban").astype(float),
            "formerly_smoked": (work["smoking_status"] == "formerly smoked").astype(float),
            "smokes": (work["smoking_status"] == "smokes").astype(float),
            "smoke_unknown": (work["smoking_status"] == "Unknown").astype(float),
        }
    )
    design = sm.add_constant(design, has_constant="add")
    fit = sm.Logit(y.to_numpy(), design.to_numpy()).fit(disp=False, maxiter=200)
    labels = {
        "age_per_10y": "Age, per 10 years",
        "glucose_per_10": "Glucose, per 10 mg/dL",
        "bmi_per_5": "BMI, per 5 points",
        "hypertension": "Hypertension",
        "heart_disease": "Heart disease",
        "married": "Ever married",
        "urban": "Urban residence",
        "formerly_smoked": "Formerly smoked",
        "smokes": "Currently smokes",
        "smoke_unknown": "Smoking unknown",
    }
    rows = []
    for name, coef, se, pval in zip(design.columns, fit.params, fit.bse, fit.pvalues):
        if name == "const":
            continue
        rows.append(
            {
                "term": labels[name],
                "odds_ratio": json_float(np.exp(coef), 3),
                "ci_low": json_float(np.exp(coef - 1.96 * se), 3),
                "ci_high": json_float(np.exp(coef + 1.96 * se), 3),
                "p_value": json_p(pval),
            }
        )
    rows = [
        row
        for row in rows
        if row["odds_ratio"] is not None and row["ci_low"] is not None and row["ci_high"] is not None
    ]
    rows.sort(key=lambda row: abs(np.log(row["odds_ratio"])), reverse=True)
    return rows


def choose_operating(grid: list[dict]) -> dict:
    """Lowest-flag threshold that still catches at least 70% of holdout strokes.

    If no grid point reaches that sensitivity, fall back to the best F1.
    """
    eligible = [row for row in grid if row["sensitivity"] >= 0.70]
    if eligible:
        return min(eligible, key=lambda row: (row["flagged"], -row["threshold"]))
    return max(grid, key=lambda row: (row["f1"], -row["flagged"]))


def per_thousand(row: dict, n: int) -> dict:
    return {
        "flagged": json_float(1000 * row["flagged"] / n, 1),
        "caught": json_float(1000 * row["tp"] / n, 1),
        "missed": json_float(1000 * row["fn"] / n, 1),
        "false_flags": json_float(1000 * row["fp"] / n, 1),
    }


def score_model(name: str, y_true, proba, *, scores_are_risks: bool) -> dict:
    return {
        "name": name,
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 3),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 3),
        "brier": round(float(brier_score_loss(y_true, proba)), 3),
        "at_half": classification_at(y_true, proba, 0.5),
        "scores_are_risks": scores_are_risks,
    }


def fit_proba(pipe: Pipeline, train_x, train_y, test_x) -> np.ndarray:
    pipe.fit(train_x, train_y)
    return pipe.predict_proba(test_x)[:, 1]


def build() -> dict:
    frame = load_frame()
    train_x, test_x, train_y, test_y = split(frame)
    y = test_y.to_numpy()

    majority_p = np.full(len(y), float(train_y.mean()))
    age_only = LogisticRegression(max_iter=400, class_weight="balanced")
    age_med = float(train_x["age"].median())
    age_only.fit(train_x[["age"]].fillna(age_med), train_y)
    age_p = age_only.predict_proba(test_x[["age"]].fillna(age_med))[:, 1]

    logistic = Pipeline(
        [
            ("pre", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=600)),
        ]
    )
    logistic_balanced = Pipeline(
        [
            ("pre", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=600, class_weight="balanced")),
        ]
    )
    tree = Pipeline(
        [
            ("pre", make_preprocessor(scale=False)),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=40,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("pre", make_preprocessor(scale=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )

    log_p = fit_proba(logistic, train_x, train_y, test_x)
    balanced_p = fit_proba(logistic_balanced, train_x, train_y, test_x)
    tree_p = fit_proba(tree, train_x, train_y, test_x)
    forest_p = fit_proba(forest, train_x, train_y, test_x)

    curves = {}
    probas = {
        "Majority rate": majority_p,
        "Age only": age_p,
        "Logistic": log_p,
        "Logistic, class-weighted": balanced_p,
        "Decision tree": tree_p,
        "Random forest": forest_p,
    }
    risk_models = {"Majority rate", "Logistic"}
    models = []
    for name, proba in probas.items():
        row = score_model(name, y, proba, scores_are_risks=name in risk_models)
        fpr, tpr, _ = roc_curve(y, proba)
        curves[name] = downsample_curve(fpr, tpr)
        models.append(row)

    # Desk probabilities are unweighted. Class weights rank well and calibrate badly.
    desk = "Logistic"
    desk_p = log_p
    grid = threshold_grid(y, desk_p)
    operating = dict(choose_operating(grid))
    n = len(y)
    operating["per_1000"] = per_thousand(operating, n)
    operating["rule"] = (
        "lowest flag count among thresholds with sensitivity >= 0.70"
        if operating["sensitivity"] >= 0.70
        else "best F1; no grid threshold reached sensitivity 0.70"
    )

    bmi_index = NUM.index("bmi")
    pipeline_median = float(
        logistic.named_steps["pre"]
        .named_transformers_["num"]
        .named_steps["impute"]
        .statistics_[bmi_index]
    )
    train_median = float(train_x["bmi"].median())
    full_median = float(frame["bmi"].median())
    bmi_missing = int(frame["bmi"].isna().sum())
    payload = {
        "dataset": {
            "rows": int(len(frame)),
            "strokes": int(frame["stroke"].sum()),
            "prevalence": round(float(frame["stroke"].mean()), 4),
            "bmi_missing": bmi_missing,
            "bmi_missing_pct": round(100 * bmi_missing / len(frame), 1),
            "train_rows": int(len(train_y)),
            "test_rows": int(len(test_y)),
            "test_strokes": int(test_y.sum()),
            "seed": SEED,
            "imputation": {
                "feature": "bmi",
                "strategy": "median",
                "fit_on": "train",
                "train_median": json_float(train_median, 6),
                "full_median": json_float(full_median, 6),
                "pipeline_statistic": json_float(pipeline_median, 6),
                "missing_rows": bmi_missing,
            },
        },
        "age_bands": age_bands(frame),
        "hypertension": rates_by(frame, "hypertension"),
        "heart_disease": rates_by(frame, "heart_disease"),
        "smoking": rates_by(frame, "smoking_status"),
        "models": models,
        "roc": curves,
        "desk_model": desk,
        "thresholds": grid,
        "operating": operating,
        "calibration": calibration(y, desk_p),
        "brier_desk": round(float(brier_score_loss(y, desk_p)), 3),
        "odds_ratios": odds_ratios(train_x, train_y),
        "odds_design": (
            "Unweighted statsmodels Logit on the training fold. "
            "Age per 10 years, glucose per 10 mg/dL, BMI per 5, hypertension, "
            "heart disease, married, urban, smoking dummies. "
            "gender == Other dropped. work_type and gender_Other omitted; "
            "those levels separated and produced infinite confidence intervals."
        ),
        "note": (
            "Educational casebook on the Kaggle stroke file. "
            "Not a diagnostic device and not validated for clinical use."
        ),
    }
    return payload


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        value = obj.item()
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value
    raise TypeError(type(obj))


def export() -> dict:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=False))
    return payload


def _print_report(result: dict) -> None:
    print(f"{'model':<28} {'ROC':>6} {'PR':>6} {'Brier':>7} {'sens@0.5':>8}")
    for model in result["models"]:
        print(
            f"{model['name']:<28} {model['roc_auc']:6.3f} {model['pr_auc']:6.3f} "
            f"{model['brier']:7.3f} {model['at_half']['sensitivity']:8.3f}"
        )
    op = result["operating"]
    burden = op["per_1000"]
    print(
        f"operating threshold {op['threshold']:.2f}  "
        f"sens {op['sensitivity']:.3f}  spec {op['specificity']:.3f}  "
        f"ppv {op['ppv']:.3f}  false_flags/true {op['false_flags_per_true']}"
    )
    print(
        "per 1000: "
        f"flagged {burden['flagged']}  caught {burden['caught']}  "
        f"missed {burden['missed']}  false flags {burden['false_flags']}"
    )
    print("odds ratios")
    for row in result["odds_ratios"][:8]:
        print(
            f"  {row['term']:<28} {row['odds_ratio']}  "
            f"({row['ci_low']}, {row['ci_high']})"
        )


if __name__ == "__main__":
    result = export()
    _print_report(result)
    print("wrote", OUT)
