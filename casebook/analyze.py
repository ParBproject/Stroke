"""Holdout stroke-risk casebook.

Train-only imputation, an unweighted age baseline, logistic odds ratios from a
separate association model, tree and forest comparison, a threshold frozen on
training out-of-fold scores, and holdout calibration. Not a clinical tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "healthcare-dataset-stroke-data.csv"
OUT = ROOT / "casebook" / "metrics.json"
SEED = 42
OOF_FOLDS = 5
N_BOOT = 2000
# Desk logistic is L2 (l1_ratio 0). C=1 is a light penalty at this sample size.
# Stated explicitly so a later sklearn default cannot silently drop it.
DESK_C = 1.0
DESK_L1_RATIO = 0.0
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


def make_logistic(*, class_weight: str | None = None) -> Pipeline:
    """L2-penalized logistic. Unweighted by default so scores stay probabilities."""
    clf = LogisticRegression(
        C=DESK_C,
        l1_ratio=DESK_L1_RATIO,
        max_iter=600,
        solver="lbfgs",
        random_state=SEED,
        class_weight=class_weight,
    )
    return Pipeline(
        [
            ("pre", make_preprocessor(scale=True)),
            ("clf", clf),
        ]
    )


def make_age_only() -> LogisticRegression:
    """Unweighted age logistic.

    A class-weighted age fit ranks the same single feature and then reports a
    Brier and a recall-at-0.50 that are not probabilities. This baseline is a
    probability, so those two numbers can sit next to the desk model.
    """
    return LogisticRegression(
        C=DESK_C,
        l1_ratio=DESK_L1_RATIO,
        max_iter=400,
        solver="lbfgs",
        random_state=SEED,
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


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """Wilson score interval. None when the denominator is zero."""
    if n <= 0:
        return None
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = z * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    point = round(phat, 4)
    lo_r = min(round(lo, 4), point)
    hi_r = max(round(hi, 4), point)
    if lo_r == 0:
        lo_r = 0.0
    if hi_r == 0:
        hi_r = 0.0
    return [lo_r, hi_r]


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
        "sensitivity_ci": wilson_interval(int(tp), int(tp + fn)),
        "specificity": round(float(spec), 4),
        "specificity_ci": wilson_interval(int(tn), int(tn + fp)),
        "ppv": round(float(ppv), 4),
        "ppv_ci": wilson_interval(int(tp), int(tp + fp)),
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


def calibration(y_true, proba, bins: int = 5) -> list[dict]:
    """Equal-count bins on predicted risk.

    Equal-width bins on [0, 1] put almost every row of this file into the
    first bin, which also contains the operating threshold. Counts are split
    on the sorted scores so that region is visible. The interval is a Wilson
    interval on the bin's event count, not a claim that the bin is precise.
    """
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(proba, dtype=float)
    order = np.argsort(p_arr, kind="mergesort")
    rows = []
    for group in np.array_split(order, bins):
        if len(group) == 0:
            continue
        pred = p_arr[group]
        obs = y_arr[group]
        events = int(obs.sum())
        n = int(len(group))
        observed = float(obs.mean()) if n else 0.0
        rows.append(
            {
                "lo": json_float(pred.min(), 4),
                "hi": json_float(pred.max(), 4),
                "n": n,
                "events": events,
                "mean_p": json_float(pred.mean(), 4),
                "observed": json_float(observed, 4),
                "observed_ci": wilson_interval(events, n),
            }
        )
    return rows


def calibration_summary(y_true, proba) -> dict:
    """Holdout calibration slope. Does not rescale the published probabilities."""
    p_arr = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p_arr / (1.0 - p_arr))
    design = sm.add_constant(logit, has_constant="add")
    fit = sm.Logit(np.asarray(y_true), design).fit(disp=False, maxiter=200)
    intercept, slope = (float(v) for v in fit.params)
    se_i, se_s = (float(v) for v in fit.bse)

    def ci(estimate: float, se: float):
        return _ci_bounds(estimate, estimate - 1.96 * se, estimate + 1.96 * se, 3)

    return {
        "scheme": "equal_count",
        "bins": 5,
        "slope": round(slope, 3),
        "slope_ci": ci(slope, se_s),
        "intercept": round(intercept, 3),
        "intercept_ci": ci(intercept, se_i),
        "fit_on": "holdout",
        "adjusts_probabilities": False,
    }


ASSOCIATION_NOTE = (
    "Separate unpenalized logit on the training fold, not the coefficients of the desk model. "
    "The desk logistic is L2-penalized (C=1) with work type and gender one-hot; those weights "
    "are not reference-category odds ratios. "
    "Covariates: age per 10 years, glucose per 10 mg/dL, BMI per 5, hypertension, heart disease, "
    "married, urban, and smoking dummies against never smoked. "
    "Work type is omitted because Never_worked has no strokes in the training fold and a dummy "
    "separates. Gender is not in this table; the one Other row is dropped."
)


def association_meta(train_x: pd.DataFrame, train_y: pd.Series) -> dict:
    """Why the odds-ratio table is not the desk model, and which level separates."""
    never = train_x["work_type"].eq("Never_worked")
    other = train_x["gender"].eq("Other")
    return {
        "family": "statsmodels Logit",
        "sample": "training fold",
        "penalized": False,
        "is_desk_model": False,
        "desk_penalty": "l2",
        "desk_C": DESK_C,
        "separation": {
            "factor": "work_type",
            "level": "Never_worked",
            "train_rows": int(never.sum()),
            "train_strokes": int(train_y.loc[never].sum()),
            "reason": "complete_separation",
        },
        "dropped_rows": {
            "rule": "gender == Other",
            "rows": int(other.sum()),
            "reason": "gender is not a term; the Other row is dropped so it is not a dummy",
        },
        "note": ASSOCIATION_NOTE,
    }


def odds_ratios(train_x: pd.DataFrame, train_y: pd.Series) -> list[dict]:
    """Unpenalized associations on the training fold. Not the desk model's coefficients."""
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
    """Lowest-flag threshold with sensitivity at least 0.70 on these rows.

    If no grid point reaches that sensitivity, fall back to the best F1.
    Pass training out-of-fold rows. Passing the holdout tunes the cutoff
    on the same labels used to report it.
    """
    eligible = [row for row in grid if row["sensitivity"] >= 0.70]
    if eligible:
        return min(eligible, key=lambda row: (row["flagged"], -row["threshold"]))
    return max(grid, key=lambda row: (row["f1"], -row["flagged"]))


def threshold_from_training(train_x: pd.DataFrame, train_y: pd.Series) -> tuple[float, dict]:
    """Freeze a cutoff from stratified out-of-fold scores on the training fold.

    Each fold fits imputation, scaling, and the logistic only on its fit rows.
    The desk model is refit later on the full training fold; this function
    never sees holdout rows.
    """
    scores = np.zeros(len(train_y), dtype=float)
    labels = train_y.to_numpy()
    cv = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=SEED)
    for fit_idx, valid_idx in cv.split(train_x, labels):
        fold = clone(make_logistic())
        fold.fit(train_x.iloc[fit_idx], train_y.iloc[fit_idx])
        scores[valid_idx] = fold.predict_proba(train_x.iloc[valid_idx])[:, 1]
    chosen = choose_operating(threshold_grid(labels, scores))
    reached = chosen["sensitivity"] >= 0.70
    rule = (
        "lowest flag count among thresholds with sensitivity >= 0.70"
        if reached
        else "best F1; no training out-of-fold threshold reached sensitivity 0.70"
    )
    record = {
        "source": "train_oof",
        "folds": OOF_FOLDS,
        "seed": SEED,
        "labeled_rows": int(len(train_y)),
        "threshold": chosen["threshold"],
        "oof_sensitivity": chosen["sensitivity"],
        "oof_specificity": chosen["specificity"],
        "oof_ppv": chosen["ppv"],
        "oof_flagged": chosen["flagged"],
        "rule": rule,
    }
    return float(chosen["threshold"]), record


def selection_note(threshold: float, *, reached: bool) -> str:
    if reached:
        how = (
            "the lowest flag count among cutoffs with out-of-fold sensitivity at least 0.70"
        )
    else:
        how = "the best out-of-fold F1, because no cutoff reached sensitivity 0.70"
    return (
        f"Probabilities are the unweighted logistic. Threshold {threshold:.2f} was frozen on "
        f"stratified {OOF_FOLDS}-fold scores from the training fold only: {how}. "
        "The holdout did not choose it. The slider reads the holdout grid, including ?t=0.15."
    )


def per_thousand(row: dict, n: int) -> dict:
    return {
        "flagged": json_float(1000 * row["flagged"] / n, 1),
        "caught": json_float(1000 * row["tp"] / n, 1),
        "missed": json_float(1000 * row["fn"] / n, 1),
        "false_flags": json_float(1000 * row["fp"] / n, 1),
    }


def _clean_round(value: float, digits: int) -> float:
    number = round(float(value), digits)
    if number == 0:
        return 0.0
    return float(number)


def _ci_bounds(point: float, lo: float, hi: float, digits: int) -> list:
    """Round an interval and keep it closed around the published point."""
    published = _clean_round(point, digits)
    return [min(_clean_round(lo, digits), published), max(_clean_round(hi, digits), published)]


def _quantile_ci(samples, point: float, digits: int) -> list:
    lo, hi = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975])
    return _ci_bounds(point, float(lo), float(hi), digits)


def bootstrap_vs_age(y_true, age_p, log_p, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired percentile intervals on the holdout. The models are not refit."""
    y_arr = np.asarray(y_true)
    age_p = np.asarray(age_p, dtype=float)
    log_p = np.asarray(log_p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_arr)
    buckets = {key: [] for key in ("age_roc", "log_roc", "age_pr", "log_pr", "age_br", "log_br", "d_roc", "d_pr", "d_br")}
    kept = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y = y_arr[idx]
        if y.min() == y.max():
            continue
        kept += 1
        a_roc = float(roc_auc_score(y, age_p[idx]))
        l_roc = float(roc_auc_score(y, log_p[idx]))
        a_pr = float(average_precision_score(y, age_p[idx]))
        l_pr = float(average_precision_score(y, log_p[idx]))
        a_br = float(brier_score_loss(y, age_p[idx]))
        l_br = float(brier_score_loss(y, log_p[idx]))
        buckets["age_roc"].append(a_roc)
        buckets["log_roc"].append(l_roc)
        buckets["age_pr"].append(a_pr)
        buckets["log_pr"].append(l_pr)
        buckets["age_br"].append(a_br)
        buckets["log_br"].append(l_br)
        buckets["d_roc"].append(l_roc - a_roc)
        buckets["d_pr"].append(l_pr - a_pr)
        buckets["d_br"].append(l_br - a_br)
    age_roc = float(roc_auc_score(y_arr, age_p))
    log_roc = float(roc_auc_score(y_arr, log_p))
    age_pr = float(average_precision_score(y_arr, age_p))
    log_pr = float(average_precision_score(y_arr, log_p))
    age_br = float(brier_score_loss(y_arr, age_p))
    log_br = float(brier_score_loss(y_arr, log_p))
    roc_lift = _clean_round(log_roc - age_roc, 3)
    pr_lift = _clean_round(log_pr - age_pr, 3)
    brier_lift = _clean_round(log_br - age_br, 3)
    return {
        "age": {
            "roc_auc_ci": _quantile_ci(buckets["age_roc"], age_roc, 3),
            "pr_auc_ci": _quantile_ci(buckets["age_pr"], age_pr, 3),
            "brier_ci": _quantile_ci(buckets["age_br"], age_br, 3),
        },
        "logistic": {
            "roc_auc_ci": _quantile_ci(buckets["log_roc"], log_roc, 3),
            "pr_auc_ci": _quantile_ci(buckets["log_pr"], log_pr, 3),
            "brier_ci": _quantile_ci(buckets["log_br"], log_br, 3),
        },
        "vs_age": {
            "reference": "Age only",
            "comparison": "Logistic",
            "resamples": n_boot,
            "resamples_kept": kept,
            "seed": seed,
            "method": "paired percentile bootstrap of holdout rows; models are not refit",
            "roc_auc": roc_lift,
            "roc_auc_ci": _ci_bounds(roc_lift, *np.quantile(buckets["d_roc"], [0.025, 0.975]), 3),
            "pr_auc": pr_lift,
            "pr_auc_ci": _ci_bounds(pr_lift, *np.quantile(buckets["d_pr"], [0.025, 0.975]), 3),
            "brier": brier_lift,
            "brier_ci": _ci_bounds(brier_lift, *np.quantile(buckets["d_br"], [0.025, 0.975]), 3),
        },
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
    age_only = make_age_only()
    age_med = float(train_x["age"].median())
    age_only.fit(train_x[["age"]].fillna(age_med), train_y)
    age_p = age_only.predict_proba(test_x[["age"]].fillna(age_med))[:, 1]

    logistic = make_logistic()
    logistic_balanced = make_logistic(class_weight="balanced")
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
    risk_models = {"Majority rate", "Age only", "Logistic"}
    models = []
    for name, proba in probas.items():
        row = score_model(name, y, proba, scores_are_risks=name in risk_models)
        fpr, tpr, _ = roc_curve(y, proba)
        curves[name] = downsample_curve(fpr, tpr)
        models.append(row)

    uncertainty = bootstrap_vs_age(y, age_p, log_p)
    by_name = {row["name"]: row for row in models}
    by_name["Age only"].update(uncertainty["age"])
    by_name["Logistic"].update(uncertainty["logistic"])

    # Desk probabilities are unweighted. Class weights rank well and calibrate badly.
    # The cutoff is frozen on training out-of-fold scores. The holdout grid is
    # the slider only; it does not choose the operating point.
    desk = "Logistic"
    desk_p = log_p
    frozen, selection = threshold_from_training(train_x, train_y)
    grid = threshold_grid(y, desk_p)
    operating = dict(classification_at(y, desk_p, frozen))
    n = len(y)
    operating["per_1000"] = per_thousand(operating, n)
    operating["selected_on"] = selection["source"]
    operating["rule"] = (
        selection["rule"] + "; chosen on training out-of-fold scores, then scored once on the holdout"
    )
    operating["selection_note"] = selection_note(
        frozen, reached=selection["oof_sensitivity"] >= 0.70
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
        "thresholds_note": (
            "Holdout classification at each cutoff of the refit unweighted logistic. "
            "Descriptive only. The operating threshold is threshold_selection.threshold, "
            "frozen before the holdout was scored. "
            "sensitivity_ci, specificity_ci, and ppv_ci are Wilson intervals on these holdout counts."
        ),
        "thresholds": grid,
        "threshold_selection": selection,
        "operating": operating,
        "vs_age": uncertainty["vs_age"],
        "calibration": calibration(y, desk_p),
        "calibration_summary": calibration_summary(y, desk_p),
        "brier_desk": round(float(brier_score_loss(y, desk_p)), 3),
        "odds_ratios": odds_ratios(train_x, train_y),
        "association_model": association_meta(train_x, train_y),
        "odds_design": ASSOCIATION_NOTE,
        "desk_fit": {
            "estimator": "LogisticRegression",
            "penalty": "l2",
            "C": DESK_C,
            "l1_ratio": DESK_L1_RATIO,
            "class_weight": None,
        },
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
    selected = result["threshold_selection"]
    print(
        f"threshold frozen on train OOF ({selected['folds']}-fold, n={selected['labeled_rows']}) "
        f"{selected['threshold']:.2f}  oof sens {selected['oof_sensitivity']:.3f}  "
        f"oof spec {selected['oof_specificity']:.3f}  oof ppv {selected['oof_ppv']:.3f}  "
        f"oof flagged {selected['oof_flagged']}"
    )
    print(
        f"holdout at frozen threshold {op['threshold']:.2f}  "
        f"sens {op['sensitivity']:.3f}  spec {op['specificity']:.3f}  "
        f"ppv {op['ppv']:.3f}  false_flags/true {op['false_flags_per_true']}"
    )
    sens = op["sensitivity_ci"]
    print(
        f"holdout sensitivity Wilson CI {sens[0]:.4f}–{sens[1]:.4f}  "
        f"ppv CI {op['ppv_ci'][0]:.4f}–{op['ppv_ci'][1]:.4f}"
    )
    lift = result["vs_age"]
    print(
        f"vs unweighted age  ROC {lift['roc_auc']:+.3f} {lift['roc_auc_ci']}  "
        f"PR {lift['pr_auc']:+.3f} {lift['pr_auc_ci']}  "
        f"Brier {lift['brier']:+.3f} {lift['brier_ci']}"
    )
    slope = result["calibration_summary"]
    print(
        f"calibration slope {slope['slope']:.3f} {slope['slope_ci']}  "
        f"intercept {slope['intercept']:.3f} (probabilities not rescaled)"
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
