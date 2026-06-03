from __future__ import annotations

import argparse

import numpy as np

from src.common.data import load_multiclass_dataset, sample_uniform
from src.common.metrics import agreement_score
from src.common.oracle import ClassifierOracle
from src.equation_solving.run_multiclass_lr import train_target
from src.label_only.retraining import adaptive_retraining, uniform_retraining


def _parse_budgets(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def evaluate_clone(
    oracle: ClassifierOracle,
    model,
    X_test: np.ndarray,
    rng: np.random.Generator,
    uniform_samples: int,
    bounds: tuple[float, float],
) -> dict[str, float]:
    X_uniform = sample_uniform(uniform_samples, oracle.n_features, rng, bounds=bounds)
    return {
        "test_agreement": agreement_score(oracle.query_label(X_test), model.predict(X_test)),
        "uniform_agreement": agreement_score(oracle.query_label(X_uniform), model.predict(X_uniform)),
    }


def run(
    dataset: str,
    target_model: str,
    budgets: list[int],
    seed: int,
    uniform_samples: int,
) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    data = load_multiclass_dataset(dataset, random_state=seed)

    target = train_target(target_model, seed)
    target.fit(data.X_train, data.y_train)
    oracle = ClassifierOracle(target, n_features=data.n_features)

    rows: list[dict[str, float | str]] = []
    for budget in budgets:
        for extractor in (uniform_retraining, adaptive_retraining):
            result = extractor(oracle, budget=budget, rng=rng, seed=seed, bounds=data.bounds)
            metrics = evaluate_clone(
                oracle,
                result.model,
                data.X_test,
                rng,
                uniform_samples=uniform_samples,
                bounds=data.bounds,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "target_model": target_model,
                    "strategy": result.strategy,
                    "budget": float(budget),
                    "extraction_queries": float(result.query_count),
                    **metrics,
                }
            )
    return rows


def _format_value(key: str, value: float | str) -> str:
    if isinstance(value, str):
        return value
    if key.endswith("agreement"):
        return f"{value:.6f}"
    return f"{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare label-only uniform and adaptive retraining attacks."
    )
    parser.add_argument("--dataset", choices=["iris", "digits"], default="digits")
    parser.add_argument("--target-model", choices=["softmax", "ovr"], default="softmax")
    parser.add_argument("--budgets", default="650,6500,26000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uniform-samples", type=int, default=10_000)
    args = parser.parse_args()

    rows = run(
        dataset=args.dataset,
        target_model=args.target_model,
        budgets=_parse_budgets(args.budgets),
        seed=args.seed,
        uniform_samples=args.uniform_samples,
    )

    for row in rows:
        print(f"[{row['dataset']} / {row['target_model']} / {row['strategy']} / budget={int(row['budget'])}]")
        for key, value in row.items():
            print(f"{key}: {_format_value(key, value)}")
        print()


if __name__ == "__main__":
    main()
