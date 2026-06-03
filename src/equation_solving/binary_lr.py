from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.data import sample_uniform


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class BinaryLogisticClone:
    coef_: np.ndarray
    intercept_: float
    classes_: np.ndarray

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_pos = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]


@dataclass(frozen=True)
class BinaryExtractionResult:
    clone: BinaryLogisticClone
    query_count: int
    matrix_rank: int
    residual_norm: float


def generate_full_rank_queries(
    n_features: int,
    rng: np.random.Generator,
    n_queries: int | None = None,
    bounds: tuple[float, float] = (-1.0, 1.0),
    max_attempts: int = 1_000,
) -> np.ndarray:
    """Generate queries whose augmented matrix [X, 1] has full column rank."""
    if n_queries is None:
        n_queries = n_features + 1
    if n_queries < n_features + 1:
        raise ValueError("binary LR extraction needs at least d + 1 queries")

    for _ in range(max_attempts):
        X = sample_uniform(n_queries, n_features, rng, bounds=bounds)
        X_aug = np.column_stack([X, np.ones(n_queries)])
        if np.linalg.matrix_rank(X_aug) == n_features + 1:
            return X

    raise RuntimeError("failed to generate a full-rank query matrix")


def extract_binary_logistic_regression(
    oracle,
    rng: np.random.Generator,
    n_queries: int | None = None,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> BinaryExtractionResult:
    """Recover a binary logistic regression from confidence values."""
    n_features = oracle.n_features
    X = generate_full_rank_queries(n_features, rng, n_queries=n_queries, bounds=bounds)

    p = oracle.query_proba(X)
    if p.shape[1] != 2:
        raise ValueError(f"expected a binary classifier, got {p.shape[1]} classes")

    y = logit(p[:, 1])
    X_aug = np.column_stack([X, np.ones(X.shape[0])])

    if X_aug.shape[0] == X_aug.shape[1]:
        theta = np.linalg.solve(X_aug, y)
    else:
        theta = np.linalg.lstsq(X_aug, y, rcond=None)[0]

    residual = X_aug @ theta - y
    clone = BinaryLogisticClone(
        coef_=theta[:-1],
        intercept_=float(theta[-1]),
        classes_=np.asarray(oracle.classes_),
    )

    return BinaryExtractionResult(
        clone=clone,
        query_count=X.shape[0],
        matrix_rank=int(np.linalg.matrix_rank(X_aug)),
        residual_norm=float(np.linalg.norm(residual)),
    )

