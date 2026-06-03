from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.common.data import load_binary_dataset, sample_uniform
from src.common.metrics import agreement_score, mean_total_variation
from src.common.oracle import ClassifierOracle
from src.equation_solving.binary_lr import extract_binary_logistic_regression


def run(seed: int, uniform_samples: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    data = load_binary_dataset(random_state=seed)

    target = LogisticRegression(max_iter=2_000, solver="lbfgs", random_state=seed)
    target.fit(data.X_train, data.y_train)

    oracle = ClassifierOracle(target, n_features=data.n_features)
    extraction = extract_binary_logistic_regression(oracle, rng, bounds=data.bounds)

    X_uniform = sample_uniform(uniform_samples, data.n_features, rng, bounds=data.bounds)

    y_test_oracle = oracle.query_label(data.X_test)
    y_test_clone = extraction.clone.predict(data.X_test)
    p_test_oracle = oracle.query_proba(data.X_test)
    p_test_clone = extraction.clone.predict_proba(data.X_test)

    y_uniform_oracle = oracle.query_label(X_uniform)
    y_uniform_clone = extraction.clone.predict(X_uniform)
    p_uniform_oracle = oracle.query_proba(X_uniform)
    p_uniform_clone = extraction.clone.predict_proba(X_uniform)

    return {
        "features": float(data.n_features),
        "classes": float(len(oracle.classes_)),
        "extraction_queries": float(extraction.query_count),
        "matrix_rank": float(extraction.matrix_rank),
        "residual_norm": extraction.residual_norm,
        "test_agreement": agreement_score(y_test_oracle, y_test_clone),
        "uniform_agreement": agreement_score(y_uniform_oracle, y_uniform_clone),
        "test_tv": mean_total_variation(p_test_oracle, p_test_clone),
        "uniform_tv": mean_total_variation(p_uniform_oracle, p_uniform_clone),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a binary logistic regression with d + 1 confidence queries."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uniform-samples", type=int, default=10_000)
    args = parser.parse_args()

    result = run(seed=args.seed, uniform_samples=args.uniform_samples)
    for key, value in result.items():
        if key.endswith("agreement"):
            print(f"{key}: {value:.6f}")
        elif key.endswith("tv") or key == "residual_norm":
            print(f"{key}: {value:.3e}")
        else:
            print(f"{key}: {value:g}")


if __name__ == "__main__":
    main()

