"""Custom scikit-learn transformers used in the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierTrans(BaseEstimator, TransformerMixin):
    """Replace values outside the IQR fences (Q1 - t*IQR, Q3 + t*IQR) with NaN.

    Fences are computed on the data being transformed, as in the original notebook.
    """

    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = pd.DataFrame(X).copy()
        for col in X_new.columns:
            q3, q1 = np.nanpercentile(X_new[col].astype(float), [75, 25])
            iqr = self.threshold * (q3 - q1)
            mask = (X_new[col] > q3 + iqr) | (X_new[col] < q1 - iqr)
            X_new.loc[mask, col] = np.nan
        return X_new.values


class MakeOrdinal(BaseEstimator, TransformerMixin):
    """Map each column's sorted unique values to 0, 1, 2, ..."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = pd.DataFrame(X).copy()
        for col in X_new.columns:
            cats = np.sort(X_new[col].unique())
            X_new[col] = np.searchsorted(cats, X_new[col].to_numpy())
        return X_new.values


class ReplaceVals(BaseEstimator, TransformerMixin):
    """Apply a pandas `replace` mapping, e.g. {column: {label: number}}."""

    def __init__(self, repdict: dict):
        self.repdict = repdict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = pd.DataFrame(X).copy()
        for col, mapping in self.repdict.items():
            if col in X_new.columns:
                X_new[col] = X_new[col].map(lambda v, m=mapping: m.get(v, v))
        return X_new.values.astype(float)
