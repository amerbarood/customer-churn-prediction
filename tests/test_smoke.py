"""Fast checks on a sample of the real data."""

import numpy as np
import pandas as pd
import pytest

from churn import data, model
from churn.transformers import MakeOrdinal, OutlierTrans, ReplaceVals


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return data.load().sample(3000, random_state=0)


def test_cleaning(sample):
    assert "Error" not in sample["avg_frequency_login_days"].astype(str).values
    assert "?" not in sample["joined_through_referral"].values
    assert sample["age_group"].notna().all()


def test_transformers():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 2.0, 100.0]})
    assert np.isnan(OutlierTrans(1.5).fit_transform(X)[-1, 0])
    assert MakeOrdinal().fit_transform(pd.DataFrame({"g": ["b", "a", "c"]})).ravel().tolist() == [1, 0, 2]
    out = ReplaceVals({"m": {"x": 0, "y": 1}}).fit_transform(pd.DataFrame({"m": ["y", "x"]}))
    assert out.ravel().tolist() == [1.0, 0.0]


def test_pipeline_fits(sample):
    X_train, X_test, y_train, y_test = model.split(sample)
    pipe = model.build_pipeline(max_depth=3, random_state=0).fit(X_train, y_train)
    metrics = model.evaluate(pipe, X_test, y_test)
    assert metrics.accuracy > 0.8
    assert len(model.feature_importances(pipe)) > 0
