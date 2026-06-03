from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np

from src.common.data import load_multiclass_dataset, sample_uniform
from src.common.metrics import agreement_score, mean_total_variation
from src.common.oracle import ClassifierOracle, RoundedProbaOracle
from src.equation_solving.multiclass_lr import (
    extract_ovr_regression,
    extract_softmax_regression,
    parameter_count,
)
from src.equation_solving.run_multiclass_lr import train_target


def _parse_decimals(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _iter_choices(value: str, choices: Iterable[str]) -> list[str]:
    choices = list(choices)
    return choices if value == "all" else [value]


def run_one(
    dataset: str,
    model_type: str,
    decimals: int,
    seed: int,
    budget_multiplier: float,
    uniform_samples: int,
) -> dict[str, float | str | bool]:
    rng = np.random.default_rng(seed)
    data = load_multiclass_dataset(dataset, random_state=seed)

    target = train_target(model_type, seed)
    target.fit(data.X_train, data.y_train)

    true_oracle = ClassifierOracle(target, n_features=data.n_features)
    attack_oracle = RoundedProbaOracle(true_oracle, decimals=decimals)

    if model_type == "softmax":
        extraction = extract_softmax_regression(
            attack_oracle,
            rng,
            budget_multiplier=budget_multiplier,
            bounds=data.bounds,
        )
    else:
        extraction = extract_ovr_regression(
            attack_oracle,
            rng,
            budget_multiplier=budget_multiplier,
            bounds=data.bounds,
        )

    X_uniform = sample_uniform(uniform_samples, data.n_features, rng, bounds=data.bounds)

    y_test_true = true_oracle.query_label(data.X_test)
    y_test_clone = extraction.clone.predict(data.X_test)
    p_test_true = true_oracle.query_proba(data.X_test)
    p_test_clone = extraction.clone.predict_proba(data.X_test)

    y_uniform_true = true_oracle.query_label(X_uniform)
    y_uniform_clone = extraction.clone.predict(X_uniform)
    p_uniform_true = true_oracle.query_proba(X_uniform)
    p_uniform_clone = extraction.clone.predict_proba(X_uniform)

    return {
        "dataset": dataset,
        "model": model_type,
        "decimals": float(decimals),
        "features": float(data.n_features),
        "classes": float(len(true_oracle.classes_)),
        "parameters": float(parameter_count(data.n_features, len(true_oracle.classes_))),
        "budget_multiplier": budget_multiplier,
        "extraction_queries": float(extraction.query_count),
        "optimizer_success": extraction.optimizer_success,
        "optimizer_iterations": float(extraction.n_iterations),
        "test_agreement": agreement_score(y_test_true, y_test_clone),
        "uniform_agreement": agreement_score(y_uniform_true, y_uniform_clone),
        "test_tv": mean_total_variation(p_test_true, p_test_clone),
        "uniform_tv": mean_total_variation(p_uniform_true, p_uniform_clone),
    }


def _format_value(key: str, value: float | str | bool) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    if key.endswith("agreement"):
        return f"{value:.6f}"
    if key.endswith("tv"):
        return f"{value:.3e}"
    return f"{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate confidence rounding against equation-solving extraction."
    )
    parser.add_argument("--dataset", choices=["all", "iris", "digits"], default="digits")
    parser.add_argument("--model", choices=["all", "softmax", "ovr"], default="softmax")
    parser.add_argument("--decimals", default="5,4,3,2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-multiplier", type=float, default=1.0)
    parser.add_argument("--uniform-samples", type=int, default=10_000)
    args = parser.parse_args()

    datasets = _iter_choices(args.dataset, ["iris", "digits"])
    models = _iter_choices(args.model, ["softmax", "ovr"])

    for dataset in datasets:
        for model_type in models:
            for decimals in _parse_decimals(args.decimals):
                result = run_one(
                    dataset=dataset,
                    model_type=model_type,
                    decimals=decimals,
                    seed=args.seed,
                    budget_multiplier=args.budget_multiplier,
                    uniform_samples=args.uniform_samples,
                )
                print(f"[{dataset} / {model_type} / round={decimals}]")
                for key, value in result.items():
                    print(f"{key}: {_format_value(key, value)}")
                print()


if __name__ == "__main__":
    main()
