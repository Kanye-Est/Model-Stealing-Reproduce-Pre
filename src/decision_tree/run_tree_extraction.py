from __future__ import annotations

import argparse

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.common.data import load_multiclass_dataset
from src.decision_tree.path_finding import (
    DecisionTreeOracle,
    evaluate_extracted_tree,
    extract_tree_paths,
)


def run(
    seed: int,
    dataset: str,
    max_depth: int,
    epsilon: float,
    uniform_samples: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    data = load_multiclass_dataset(dataset, random_state=seed)

    target = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    target.fit(data.X_train, data.y_train)

    oracle = DecisionTreeOracle(target, n_features=data.n_features)
    extracted = extract_tree_paths(
        oracle=oracle,
        n_features=data.n_features,
        bounds=data.bounds,
        epsilon=epsilon,
    )
    metrics = evaluate_extracted_tree(
        oracle=oracle,
        extracted=extracted,
        X_test=data.X_test,
        rng=rng,
        uniform_samples=uniform_samples,
        bounds=data.bounds,
    )

    return {
        "features": float(data.n_features),
        "target_leaves": float(target.get_n_leaves()),
        "extracted_leaves": float(extracted.n_leaves),
        "max_depth": float(target.get_depth()),
        "epsilon": epsilon,
        "extraction_queries": float(extracted.query_count),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a local sklearn decision tree with bottom-up path finding."
    )
    parser.add_argument("--dataset", choices=["iris", "digits"], default="iris")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--uniform-samples", type=int, default=10_000)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        dataset=args.dataset,
        max_depth=args.max_depth,
        epsilon=args.epsilon,
        uniform_samples=args.uniform_samples,
    )
    for key, value in result.items():
        if key.endswith("agreement"):
            print(f"{key}: {value:.6f}")
        elif key == "epsilon":
            print(f"{key}: {value:.1e}")
        else:
            print(f"{key}: {value:g}")


if __name__ == "__main__":
    main()
