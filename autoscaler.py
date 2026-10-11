import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class AutoScaler(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(float).replace([np.inf, -np.inf], np.nan)
        self.fill_ = X.max().fillna(0.0) + 1
        X = X.fillna(self.fill_)

        nonneg = (X >= 0).all()
        skew = X.skew().abs()
        self.log_cols_ = X.columns[(skew > self.skew_threshold) & nonneg].tolist()

        X = self._log(X)
        self.scaler_ = StandardScaler().fit(X)
        self.columns_ = X.columns
        return self

    def _log(self, X):
        X = X.copy()
        if self.log_cols_:
            X[self.log_cols_] = np.log1p(X[self.log_cols_].clip(lower=0))
        return X

    def transform(self, X):
        X = pd.DataFrame(X).astype(float)
        X.columns = self.columns_
        X = X.replace([np.inf, -np.inf], np.nan).fillna(self.fill_)
        X = self._log(X)
        return self.scaler_.transform(X).astype(np.float32)