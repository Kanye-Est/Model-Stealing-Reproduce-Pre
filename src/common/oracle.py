from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QueryCounter:
    count: int = 0

    def add(self, X: np.ndarray) -> None:
        self.count += int(np.asarray(X).shape[0])

    def reset(self) -> None:
        self.count = 0


class ClassifierOracle:
    """Black-box wrapper exposing only prediction queries."""

    def __init__(self, model, n_features: int):
        self.model = model
        self.n_features = n_features
        self.counter = QueryCounter()

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_

    @property
    def query_count(self) -> int:
        return self.counter.count

    def reset_count(self) -> None:
        self.counter.reset()

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        return X

    def query_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._check_X(X)
        self.counter.add(X)
        return self.model.predict_proba(X)

    def query_label(self, X: np.ndarray) -> np.ndarray:
        X = self._check_X(X)
        self.counter.add(X)
        return self.model.predict(X)


class RoundedProbaOracle:
    """Oracle wrapper that rounds confidence values before returning them."""

    def __init__(self, base_oracle: ClassifierOracle, decimals: int):
        self.base_oracle = base_oracle
        self.decimals = decimals
        self.n_features = base_oracle.n_features

    @property
    def classes_(self) -> np.ndarray:
        return self.base_oracle.classes_

    @property
    def query_count(self) -> int:
        return self.base_oracle.query_count

    def reset_count(self) -> None:
        self.base_oracle.reset_count()

    def query_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.base_oracle.query_proba(X)
        rounded = np.round(probas, self.decimals)
        row_sums = np.sum(rounded, axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] == 0.0
        if np.any(zero_rows):
            rounded[zero_rows] = 1.0 / rounded.shape[1]
            row_sums = np.sum(rounded, axis=1, keepdims=True)
        return rounded / row_sums

    def query_label(self, X: np.ndarray) -> np.ndarray:
        return self.base_oracle.query_label(X)
