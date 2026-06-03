from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class RBFRepresenterClassifier:
    """Small RBF representer classifier for illustrating KLR-style leakage."""

    def __init__(self, representers: np.ndarray, labels: np.ndarray, gamma: float = 0.08):
        self.representers = np.asarray(representers, dtype=float)
        self.labels = np.asarray(labels)
        self.gamma = gamma
        self.classes_ = np.unique(labels)

    def _kernel(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        sq_dists = np.sum((X[:, None, :] - self.representers[None, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * sq_dists)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        K = self._kernel(X)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            scores[:, idx] = np.sum(K[:, self.labels == cls], axis=1)
        scores += 1e-12
        return scores / np.sum(scores, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def choose_representers(
    X: np.ndarray,
    y: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    reps = []
    labels = []
    for cls in np.unique(y):
        indices = np.flatnonzero(y == cls)
        chosen = rng.choice(indices, size=per_class, replace=False)
        reps.append(X[chosen])
        labels.append(y[chosen])
    return np.vstack(reps), np.concatenate(labels)


def plot_leakage(
    representers: np.ndarray,
    labels: np.ndarray,
    output: Path,
    max_per_class: int,
) -> None:
    classes = np.unique(labels)
    fig, axes = plt.subplots(len(classes), max_per_class, figsize=(max_per_class * 1.2, len(classes) * 1.2))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row, cls in enumerate(classes):
        class_reps = representers[labels == cls][:max_per_class]
        for col in range(max_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(class_reps):
                ax.imshow(class_reps[col].reshape(8, 8), cmap="gray_r", interpolation="nearest")
                if col == 0:
                    ax.set_ylabel(str(cls), rotation=0, labelpad=10, va="center")

    fig.suptitle("Leaked KLR representers: training images embedded in the extracted model", fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def run(seed: int, per_class: int, gamma: float, output: Path) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.30,
        random_state=seed,
        stratify=digits.target,
    )

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    representers, labels = choose_representers(X_train, y_train, per_class=per_class, rng=rng)
    oracle = RBFRepresenterClassifier(representers, labels, gamma=gamma)

    # In KLR-style models, extracting the white-box model exposes the representer matrix.
    extracted_representers = np.array(oracle.representers, copy=True)
    extracted_labels = np.array(oracle.labels, copy=True)
    plot_leakage(extracted_representers, extracted_labels, output=output, max_per_class=per_class)

    return {
        "classes": float(len(oracle.classes_)),
        "representers_per_class": float(per_class),
        "total_representers": float(len(extracted_representers)),
        "gamma": gamma,
        "test_accuracy": float(accuracy_score(y_test, oracle.predict(X_test))),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize training-data leakage from a KLR-style representer model."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=Path("figures/klr_leakage.png"))
    args = parser.parse_args()

    result = run(seed=args.seed, per_class=args.per_class, gamma=args.gamma, output=args.output)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:g}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
