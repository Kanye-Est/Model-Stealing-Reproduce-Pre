from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    bounds: tuple[float, float] = (-1.0, 1.0)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def load_binary_dataset(random_state: int = 0) -> DatasetSplit:
    """Load a small binary dataset and scale features into [-1, 1]."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.30,
        random_state=random_state,
        stratify=data.target,
    )

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(data.feature_names),
    )


def load_multiclass_dataset(name: str, random_state: int = 0) -> DatasetSplit:
    """Load a multiclass sklearn dataset and scale features into [-1, 1]."""
    loaders = {
        "iris": load_iris,
        "digits": load_digits,
    }
    if name not in loaders:
        supported = ", ".join(sorted(loaders))
        raise ValueError(f"unknown multiclass dataset {name!r}; expected one of: {supported}")

    data = loaders[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.30,
        random_state=random_state,
        stratify=data.target,
    )

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    feature_names = getattr(data, "feature_names", None)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(data.data.shape[1])]

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(feature_names),
    )


def sample_uniform(
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    low, high = bounds
    return rng.uniform(low, high, size=(n_samples, n_features))
