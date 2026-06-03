from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.common.data import load_multiclass_dataset, sample_uniform
from src.common.metrics import agreement_score
from src.common.oracle import ClassifierOracle
from src.decision_tree.path_finding import DecisionTreeOracle, extract_tree_paths
from src.equation_solving.run_multiclass_lr import run_one as run_multiclass_one
from src.equation_solving.run_multiclass_lr import train_target
from src.equation_solving.run_rounding import run_one as run_rounding_one
from src.klr_leakage.run_klr_leakage import run as run_klr
from src.label_only.retraining import adaptive_retraining, uniform_retraining


FIGURES = Path("figures")


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_equation_budget_curve() -> None:
    multipliers = [0.25, 0.5, 0.75, 1.0]
    rows = [
        run_multiclass_one("digits", "softmax", seed=0, budget_multiplier=m, uniform_samples=4_000)
        for m in multipliers
    ]
    queries = [row["extraction_queries"] for row in rows]
    test = [row["test_agreement"] for row in rows]
    uniform = [row["uniform_agreement"] for row in rows]

    plt.figure(figsize=(6.8, 4.0))
    plt.plot(queries, test, marker="o", label="test agreement")
    plt.plot(queries, uniform, marker="s", label="uniform agreement")
    plt.axvline(650, color="0.35", linestyle="--", linewidth=1.2, label="1 query / parameter")
    plt.ylim(0.90, 1.01)
    plt.xlabel("confidence queries")
    plt.ylabel("agreement with target")
    plt.title("Equation-solving extraction on Digits softmax")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    savefig(FIGURES / "equation_budget_curve.png")


def _evaluate_label_model(oracle: ClassifierOracle, model, X_test: np.ndarray, X_uniform: np.ndarray) -> tuple[float, float]:
    return (
        agreement_score(oracle.query_label(X_test), model.predict(X_test)),
        agreement_score(oracle.query_label(X_uniform), model.predict(X_uniform)),
    )


def plot_confidence_vs_label_only() -> None:
    rng = np.random.default_rng(0)
    data = load_multiclass_dataset("digits", random_state=0)
    target = train_target("softmax", seed=0)
    target.fit(data.X_train, data.y_train)
    oracle = ClassifierOracle(target, n_features=data.n_features)
    X_uniform = sample_uniform(4_000, data.n_features, rng, bounds=data.bounds)

    budgets = [650, 6_500, 26_000]
    uniform_scores = []
    adaptive_scores = []
    for budget in budgets:
        uniform = uniform_retraining(oracle, budget=budget, rng=rng, seed=0, bounds=data.bounds)
        adaptive = adaptive_retraining(oracle, budget=budget, rng=rng, seed=0, bounds=data.bounds)
        uniform_scores.append(_evaluate_label_model(oracle, uniform.model, data.X_test, X_uniform)[1])
        adaptive_scores.append(_evaluate_label_model(oracle, adaptive.model, data.X_test, X_uniform)[1])

    plt.figure(figsize=(6.8, 4.0))
    plt.plot(budgets, uniform_scores, marker="o", label="label-only uniform")
    plt.plot(budgets, adaptive_scores, marker="s", label="label-only adaptive")
    plt.scatter([650], [1.0], marker="*", s=180, color="#c43c2f", label="confidence equation-solving")
    plt.xscale("log")
    plt.ylim(0.60, 1.02)
    plt.xlabel("queries (log scale)")
    plt.ylabel("uniform agreement")
    plt.title("Confidence values vs label-only extraction")
    plt.grid(alpha=0.25, which="both")
    plt.legend(loc="lower right")
    savefig(FIGURES / "confidence_vs_label_only.png")


def plot_rounding_curve() -> None:
    decimals = [5, 4, 3, 2]
    rows = [
        run_rounding_one("digits", "softmax", decimals=d, seed=0, budget_multiplier=1.0, uniform_samples=4_000)
        for d in decimals
    ]
    agreement = [row["uniform_agreement"] for row in rows]
    tv = [row["uniform_tv"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.0))
    ax1.plot(decimals, agreement, marker="o", color="#1f77b4", label="uniform agreement")
    ax1.set_xlabel("reported confidence decimals")
    ax1.set_ylabel("agreement", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0.985, 1.001)
    ax1.invert_xaxis()
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(decimals, tv, marker="s", color="#c43c2f", label="TV distance")
    ax2.set_ylabel("mean TV distance", color="#c43c2f")
    ax2.tick_params(axis="y", labelcolor="#c43c2f")
    ax2.set_yscale("log")
    ax2.set_title("Effect of confidence rounding")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right")
    savefig(FIGURES / "rounding_curve.png")


def plot_tree_path_finding() -> None:
    data = load_multiclass_dataset("iris", random_state=0)
    target = DecisionTreeClassifier(max_depth=3, random_state=0)
    target.fit(data.X_train, data.y_train)
    oracle = DecisionTreeOracle(target, n_features=data.n_features)
    extracted = extract_tree_paths(oracle, n_features=data.n_features, bounds=data.bounds, epsilon=1e-5)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    plot_tree(
        target,
        feature_names=[name.replace(" (cm)", "") for name in data.feature_names],
        class_names=[str(c) for c in target.classes_],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=axes[0],
    )
    axes[0].set_title("Target sklearn tree")

    axes[1].axis("off")
    lines = []
    for leaf in extracted.leaves:
        compact = []
        for idx, interval in enumerate(leaf.intervals):
            if interval.low > -0.999 or interval.high < 0.999:
                compact.append(f"x{idx}: [{interval.low:.2f}, {interval.high:.2f}]")
        if not compact:
            compact = ["all features"]
        lines.append(f"leaf {leaf.leaf_id} -> class {leaf.prediction}\n  " + "; ".join(compact))
    axes[1].text(
        0.02,
        0.98,
        "\n\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#999999"),
    )
    axes[1].set_title("Extracted leaf intervals")
    savefig(FIGURES / "tree_path_finding.png")


def plot_tree_leaf_id_mechanism() -> None:
    """Draw a slide-friendly schematic for the tree path-finding edge case."""
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.15, 1.05)
    ax.axis("off")

    bands = [
        (-1.0, -0.35, "#e8f1fb", "leaf A"),
        (-0.35, 0.18, "#fde9e5", "leaf B"),
        (0.18, 1.0, "#e8f1fb", "leaf A"),
    ]
    for low, high, color, label in bands:
        ax.axvspan(low, high, ymin=0.28, ymax=0.62, color=color, ec="#3c4b5f", lw=1.4)
        ax.text((low + high) / 2, 0.52, label, ha="center", va="center", fontsize=15, weight="bold")

    for x, text in [(-1.0, "left endpoint"), (1.0, "right endpoint")]:
        ax.plot([x], [0.35], marker="o", color="#1f4e79", markersize=9)
        ax.text(x, 0.16, text, ha="center", va="center", fontsize=11)

    ax.plot([-0.35, -0.35], [0.25, 0.78], color="#c43c2f", linestyle="--", lw=2)
    ax.plot([0.18, 0.18], [0.25, 0.78], color="#c43c2f", linestyle="--", lw=2)
    ax.text(-0.35, 0.86, "hidden split", ha="center", fontsize=11, color="#c43c2f")
    ax.text(0.18, 0.86, "hidden split", ha="center", fontsize=11, color="#c43c2f")

    ax.annotate(
        "endpoints can have the same leaf id",
        xy=(0.0, 0.35),
        xytext=(0.0, 0.02),
        ha="center",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="#3c4b5f", lw=1.5),
    )
    ax.text(
        0.0,
        0.98,
        "Implementation note: recover predicates along the current path, not only interval endpoints",
        ha="center",
        va="top",
        fontsize=14,
        weight="bold",
    )
    savefig(FIGURES / "tree_leaf_id_mechanism.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    plot_equation_budget_curve()
    plot_confidence_vs_label_only()
    plot_rounding_curve()
    plot_tree_path_finding()
    plot_tree_leaf_id_mechanism()
    run_klr(seed=0, per_class=5, gamma=0.08, output=FIGURES / "klr_leakage.png")
    print("Generated reproduction figures in figures/")


if __name__ == "__main__":
    main()
