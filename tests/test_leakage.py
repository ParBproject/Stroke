"""Leakage checks that do not refit the forest.

Imputation, scaling, and one-hot encoding stay inside a pipeline fit on the
training fold. The operating threshold is a training out-of-fold choice.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import casebook.analyze as analyze
from casebook.analyze import (
    CAT,
    NUM,
    load_frame,
    make_preprocessor,
    split,
    threshold_from_training,
)


def _frame(n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(50, 10, n),
            "avg_glucose_level": rng.normal(110, 20, n),
            "bmi": rng.normal(28, 4, n),
            "hypertension": rng.integers(0, 2, n),
            "heart_disease": rng.integers(0, 2, n),
            "gender": rng.choice(["Male", "Female"], n),
            "ever_married": rng.choice(["Yes", "No"], n),
            "work_type": rng.choice(["Private", "Self-employed"], n),
            "Residence_type": rng.choice(["Urban", "Rural"], n),
            "smoking_status": rng.choice(["never smoked", "smokes"], n),
        }
    )


def test_split_keeps_every_row_once():
    frame = load_frame()
    train_x, test_x, train_y, test_y = split(frame)
    assert len(train_x) + len(test_x) == len(frame)
    assert set(train_x.index).isdisjoint(test_x.index)
    assert not train_x.index.duplicated().any()
    assert int(train_y.sum() + test_y.sum()) == int(frame["stroke"].sum())


def test_training_source_does_not_resample():
    text = inspect.getsource(analyze)
    for token in ("SMOTE", "imblearn", "RandomOverSampler", "ADASYN"):
        assert token not in text


def test_preprocessor_is_unfitted_until_fit():
    pre = make_preprocessor(scale=True)
    with pytest.raises(NotFittedError):
        pre.transform(_frame())


def test_median_imputer_does_not_see_holdout_bmi():
    train = _frame()
    train["bmi"] = [10.0, 10.0, 10.0, np.nan, 10.0, 10.0, 10.0, 10.0]
    test = _frame(seed=1)
    test["bmi"] = 100.0
    pre = make_preprocessor(scale=True)
    pre.fit(train)
    stat = pre.named_transformers_["num"].named_steps["impute"].statistics_[NUM.index("bmi")]
    assert stat == pytest.approx(10.0)

    leaked = make_preprocessor(scale=True)
    leaked.fit(pd.concat([train, test], axis=0))
    leaked_stat = leaked.named_transformers_["num"].named_steps["impute"].statistics_[
        NUM.index("bmi")
    ]
    assert leaked_stat == pytest.approx(100.0)
    assert leaked_stat != pytest.approx(stat)


def test_scaler_and_encoder_match_the_training_fold_only():
    frame = load_frame()
    train_x, test_x, _, _ = split(frame)
    pre = make_preprocessor(scale=True)
    pre.fit(train_x)
    numeric = pre.named_transformers_["num"]
    stats = numeric.named_steps["impute"].statistics_
    bmi_index = NUM.index("bmi")
    assert stats[bmi_index] == pytest.approx(float(train_x["bmi"].median()))
    assert stats[bmi_index] != pytest.approx(float(frame["bmi"].median()))

    imputed = train_x[NUM].copy()
    for column, stat in zip(NUM, stats):
        imputed[column] = imputed[column].fillna(stat)
    np.testing.assert_allclose(
        numeric.named_steps["scale"].mean_,
        imputed.mean().to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    full = pd.concat([train_x, test_x], axis=0)
    leaked = make_preprocessor(scale=True)
    leaked.fit(full)
    leaked_mean = leaked.named_transformers_["num"].named_steps["scale"].mean_
    assert not np.allclose(numeric.named_steps["scale"].mean_, leaked_mean)

    encoder = pre.named_transformers_["cat"].named_steps["oh"]
    for index, column in enumerate(CAT):
        fitted = {str(level) for level in encoder.categories_[index]}
        assert fitted == set(train_x[column].astype(str).unique())


def test_holdout_only_category_is_not_in_the_encoder():
    train = _frame()
    test = train.iloc[[0]].copy()
    test["work_type"] = "HoldoutOnly"
    pre = make_preprocessor(scale=True)
    pre.fit(train)
    encoder = pre.named_transformers_["cat"].named_steps["oh"]
    work_levels = {str(level) for level in encoder.categories_[CAT.index("work_type")]}
    assert "HoldoutOnly" not in work_levels
    transformed = pre.transform(test)
    assert transformed.shape == (1, transformed.shape[1])
    assert np.isfinite(transformed).all()


def test_threshold_function_sees_only_the_training_fold(payload):
    frame = load_frame()
    train_x, test_x, train_y, test_y = split(frame)
    threshold, selection = threshold_from_training(train_x, train_y)
    assert selection["labeled_rows"] == len(train_y)
    assert selection["labeled_rows"] != len(test_y)
    assert set(train_x.index).isdisjoint(test_x.index)
    assert threshold == payload["operating"]["threshold"]
    assert selection["source"] == "train_oof"
    assert 0.02 <= threshold <= 0.50
