from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from src.common.data import sample_uniform


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


@dataclass(frozen=True)
class MulticlassLogisticClone:
    coef_: np.ndarray
    intercept_: np.ndarray
    classes_: np.ndarray
    mode: str

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if self.mode == "softmax":
            return softmax(scores)
        if self.mode == "ovr":
            probas = expit(scores)
            return probas / np.sum(probas, axis=1, keepdims=True)
        raise ValueError(f"unknown clone mode {self.mode!r}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[indices]


@dataclass(frozen=True)
class MulticlassExtractionResult:
    clone: MulticlassLogisticClone
    query_count: int
    n_parameters: int
    loss: float
    optimizer_success: bool
    optimizer_message: str
    n_iterations: int


def parameter_count(n_features: int, n_classes: int) -> int:
    return n_classes * (n_features + 1)


def generate_query_set(
    n_features: int,
    n_classes: int,
    rng: np.random.Generator,
    budget_multiplier: float,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    n_parameters = parameter_count(n_features, n_classes)
    n_queries = max(n_features + 1, int(np.ceil(budget_multiplier * n_parameters)))
    return sample_uniform(n_queries, n_features, rng, bounds=bounds)


def _unpack(theta: np.ndarray, n_features: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    params = theta.reshape(n_classes, n_features + 1)
    return params[:, :n_features], params[:, -1]


def _pack(coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.column_stack([coef, intercept]).ravel()


def _softmax_closed_form_init(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve log(p_i / p_ref) linear systems for a softmax model."""
    X_aug = np.column_stack([X, np.ones(X.shape[0])])
    Y = np.clip(Y, 1e-12, 1.0)

    n_classes = Y.shape[1]
    theta = np.zeros((n_classes, X_aug.shape[1]))
    ref = n_classes - 1
    for cls in range(n_classes - 1):
        target = np.log(Y[:, cls] / Y[:, ref])
        theta[cls] = np.linalg.lstsq(X_aug, target, rcond=None)[0]

    return theta[:, :-1], theta[:, -1]


def _softmax_loss_and_grad(
    theta: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    n_features: int,
    n_classes: int,
    alpha: float,
) -> tuple[float, np.ndarray]:
    coef, intercept = _unpack(theta, n_features, n_classes)
    logits = X @ coef.T + intercept
    P = softmax(logits)
    P_safe = np.clip(P, 1e-15, 1.0)

    loss = -float(np.sum(Y * np.log(P_safe)) / X.shape[0])
    loss += 0.5 * alpha * float(np.sum(coef * coef))

    dlogits = (P - Y) / X.shape[0]
    grad_coef = dlogits.T @ X + alpha * coef
    grad_intercept = np.sum(dlogits, axis=0)
    return loss, _pack(grad_coef, grad_intercept)


def _ovr_loss_and_grad(
    theta: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    n_features: int,
    n_classes: int,
    alpha: float,
) -> tuple[float, np.ndarray]:
    coef, intercept = _unpack(theta, n_features, n_classes)
    logits = X @ coef.T + intercept
    sigmoids = expit(logits)
    denom = np.sum(sigmoids, axis=1, keepdims=True)
    P = sigmoids / denom
    P_safe = np.clip(P, 1e-15, 1.0)

    loss = -float(np.sum(Y * np.log(P_safe)) / X.shape[0])
    loss += 0.5 * alpha * float(np.sum(coef * coef))

    dlogits = (P - Y) * (1.0 - sigmoids) / X.shape[0]
    grad_coef = dlogits.T @ X + alpha * coef
    grad_intercept = np.sum(dlogits, axis=0)
    return loss, _pack(grad_coef, grad_intercept)


def _optimize_clone(
    X: np.ndarray,
    Y: np.ndarray,
    classes: np.ndarray,
    mode: str,
    initial_theta: np.ndarray,
    alpha: float,
    max_iter: int,
) -> MulticlassExtractionResult:
    n_features = X.shape[1]
    n_classes = Y.shape[1]
    loss_grad = _softmax_loss_and_grad if mode == "softmax" else _ovr_loss_and_grad

    result = minimize(
        fun=lambda theta: loss_grad(theta, X, Y, n_features, n_classes, alpha)[0],
        x0=initial_theta,
        jac=lambda theta: loss_grad(theta, X, Y, n_features, n_classes, alpha)[1],
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": 1e-8, "ftol": 1e-12},
    )

    coef, intercept = _unpack(result.x, n_features, n_classes)
    clone = MulticlassLogisticClone(
        coef_=coef,
        intercept_=intercept,
        classes_=np.asarray(classes),
        mode=mode,
    )

    return MulticlassExtractionResult(
        clone=clone,
        query_count=X.shape[0],
        n_parameters=parameter_count(n_features, n_classes),
        loss=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        n_iterations=int(result.nit),
    )


def extract_softmax_regression(
    oracle,
    rng: np.random.Generator,
    budget_multiplier: float = 1.0,
    bounds: tuple[float, float] = (-1.0, 1.0),
    alpha: float = 1e-12,
    max_iter: int = 500,
) -> MulticlassExtractionResult:
    n_classes = len(oracle.classes_)
    X = generate_query_set(oracle.n_features, n_classes, rng, budget_multiplier, bounds)
    Y = oracle.query_proba(X)
    if Y.shape[1] != n_classes:
        raise ValueError(f"oracle returned {Y.shape[1]} classes, expected {n_classes}")

    coef, intercept = _softmax_closed_form_init(X, Y)
    return _optimize_clone(
        X=X,
        Y=Y,
        classes=oracle.classes_,
        mode="softmax",
        initial_theta=_pack(coef, intercept),
        alpha=alpha,
        max_iter=max_iter,
    )


def extract_ovr_regression(
    oracle,
    rng: np.random.Generator,
    budget_multiplier: float = 1.0,
    bounds: tuple[float, float] = (-1.0, 1.0),
    alpha: float = 1e-10,
    max_iter: int = 1_000,
) -> MulticlassExtractionResult:
    n_classes = len(oracle.classes_)
    X = generate_query_set(oracle.n_features, n_classes, rng, budget_multiplier, bounds)
    Y = oracle.query_proba(X)
    if Y.shape[1] != n_classes:
        raise ValueError(f"oracle returned {Y.shape[1]} classes, expected {n_classes}")

    n_features = oracle.n_features
    # A normalized OvR probability does not expose each independent sigmoid.
    # Log probabilities are still a useful deterministic starting point.
    initial_logits = np.log(np.clip(Y, 1e-12, 1.0))
    X_aug = np.column_stack([X, np.ones(X.shape[0])])
    theta = np.zeros((n_classes, n_features + 1))
    for cls in range(n_classes):
        theta[cls] = np.linalg.lstsq(X_aug, initial_logits[:, cls], rcond=None)[0]

    return _optimize_clone(
        X=X,
        Y=Y,
        classes=oracle.classes_,
        mode="ovr",
        initial_theta=theta.ravel(),
        alpha=alpha,
        max_iter=max_iter,
    )
