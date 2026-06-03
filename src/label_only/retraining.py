from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.common.data import sample_uniform


@dataclass(frozen=True)
class LabelOnlyResult:
    strategy: str
    budget: int
    query_count: int
    model: LogisticRegression


def train_clone(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    model = LogisticRegression(max_iter=2_000, random_state=seed, solver="lbfgs")
    model.fit(X, y)
    return model


def uniform_retraining(
    oracle,
    budget: int,
    rng: np.random.Generator,
    seed: int,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> LabelOnlyResult:
    X = sample_uniform(budget, oracle.n_features, rng, bounds=bounds)
    y = oracle.query_label(X)
    model = train_clone(X, y, seed)
    return LabelOnlyResult("uniform", budget, budget, model)


def adaptive_retraining(
    oracle,
    budget: int,
    rng: np.random.Generator,
    seed: int,
    rounds: int = 5,
    candidate_multiplier: int = 20,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> LabelOnlyResult:
    if rounds < 2:
        raise ValueError("adaptive retraining needs at least two rounds")

    initial = max(len(oracle.classes_) * 2, budget // rounds)
    initial = min(initial, budget)
    X_train = sample_uniform(initial, oracle.n_features, rng, bounds=bounds)
    y_train = oracle.query_label(X_train)

    spent = initial
    model = train_clone(X_train, y_train, seed)

    while spent < budget:
        step = min(max(1, budget // rounds), budget - spent)
        n_candidates = max(step * candidate_multiplier, step)
        X_candidates = sample_uniform(n_candidates, oracle.n_features, rng, bounds=bounds)

        probas = model.predict_proba(X_candidates)
        sorted_probas = np.sort(probas, axis=1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]
        chosen = np.argsort(margins)[:step]
        X_query = X_candidates[chosen]
        y_query = oracle.query_label(X_query)

        X_train = np.vstack([X_train, X_query])
        y_train = np.concatenate([y_train, y_query])
        spent += step
        model = train_clone(X_train, y_train, seed)

    return LabelOnlyResult("adaptive", budget, spent, model)
