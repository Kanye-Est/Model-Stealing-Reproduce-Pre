from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.data import sample_uniform


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def contains(self, value: float, eps: float = 1e-12) -> bool:
        return self.low - eps <= value <= self.high + eps


@dataclass(frozen=True)
class ExtractedLeaf:
    leaf_id: int
    prediction: int
    intervals: tuple[Interval, ...]

    def matches(self, x: np.ndarray) -> bool:
        return all(interval.contains(float(value)) for interval, value in zip(self.intervals, x))


@dataclass(frozen=True)
class TreeExtractionResult:
    leaves: tuple[ExtractedLeaf, ...]
    query_count: int

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)


class DecisionTreeOracle:
    """Black-box decision tree oracle exposing labels and unique leaf ids."""

    def __init__(self, model, n_features: int):
        self.model = model
        self.n_features = n_features
        self.query_count = 0

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        return X

    def query_label(self, X: np.ndarray) -> np.ndarray:
        X = self._check_X(X)
        self.query_count += X.shape[0]
        return self.model.predict(X)

    def query_leaf_id(self, X: np.ndarray) -> np.ndarray:
        X = self._check_X(X)
        self.query_count += X.shape[0]
        return self.model.apply(X)

    def query_leaf_and_label(self, x: np.ndarray) -> tuple[int, int]:
        X = self._check_X(x)
        leaf = int(self.query_leaf_id(X)[0])
        label = int(self.query_label(X)[0])
        return leaf, label


class ExtractedTree:
    def __init__(self, leaves: tuple[ExtractedLeaf, ...], fallback_label: int):
        self.leaves = leaves
        self.fallback_label = fallback_label

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for x in X:
            label = self.fallback_label
            for leaf in self.leaves:
                if leaf.matches(x):
                    label = leaf.prediction
                    break
            predictions.append(label)
        return np.asarray(predictions)


def _representative(intervals: tuple[Interval, ...]) -> np.ndarray:
    return np.asarray([(interval.low + interval.high) / 2.0 for interval in intervals])


def _bisect_boundary(
    oracle: DecisionTreeOracle,
    x_left: np.ndarray,
    x_right: np.ndarray,
    left_leaf: int,
    feature_idx: int,
    epsilon: float,
) -> float:
    low = float(x_left[feature_idx])
    high = float(x_right[feature_idx])
    base = np.array(x_left, copy=True)

    while high - low > epsilon:
        mid = (low + high) / 2.0
        probe = np.array(base, copy=True)
        probe[feature_idx] = mid
        mid_leaf = int(oracle.query_leaf_id(probe)[0])
        if mid_leaf == left_leaf:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _split_region_on_feature(
    oracle: DecisionTreeOracle,
    intervals: tuple[Interval, ...],
    feature_idx: int,
    epsilon: float,
) -> list[tuple[int, Interval]]:
    base = _representative(intervals)
    feature_interval = intervals[feature_idx]
    low = feature_interval.low
    high = feature_interval.high

    probe_low = np.array(base, copy=True)
    probe_high = np.array(base, copy=True)
    probe_low[feature_idx] = low
    probe_high[feature_idx] = high

    leaf_low = int(oracle.query_leaf_id(probe_low)[0])
    leaf_high = int(oracle.query_leaf_id(probe_high)[0])

    if leaf_low == leaf_high:
        return [(leaf_low, feature_interval)]

    boundary = _bisect_boundary(
        oracle=oracle,
        x_left=probe_low,
        x_right=probe_high,
        left_leaf=leaf_low,
        feature_idx=feature_idx,
        epsilon=epsilon,
    )

    left = Interval(low=low, high=boundary)
    right = Interval(low=min(boundary + epsilon, high), high=high)
    return [(leaf_low, left), (leaf_high, right)]


def _leaf_at(
    oracle: DecisionTreeOracle,
    base: np.ndarray,
    feature_idx: int,
    value: float,
) -> int:
    probe = np.array(base, copy=True)
    probe[feature_idx] = value
    return int(oracle.query_leaf_id(probe)[0])


def _discover_feature_segments(
    oracle: DecisionTreeOracle,
    base: np.ndarray,
    feature_idx: int,
    interval: Interval,
    epsilon: float,
) -> list[tuple[int, Interval]]:
    """Find leaf-id-constant intervals along one feature line."""

    def recurse(low: float, high: float, low_leaf: int, high_leaf: int) -> list[tuple[int, Interval]]:
        if high - low <= epsilon:
            return [(low_leaf, Interval(low, high))]

        mid = (low + high) / 2.0
        mid_leaf = _leaf_at(oracle, base, feature_idx, mid)

        if low_leaf == mid_leaf == high_leaf:
            return [(low_leaf, Interval(low, high))]

        return recurse(low, mid, low_leaf, mid_leaf) + recurse(mid + epsilon, high, mid_leaf, high_leaf)

    low_leaf = _leaf_at(oracle, base, feature_idx, interval.low)
    high_leaf = _leaf_at(oracle, base, feature_idx, interval.high)
    raw_segments = recurse(interval.low, interval.high, low_leaf, high_leaf)

    merged: list[tuple[int, Interval]] = []
    for leaf_id, segment in raw_segments:
        if merged and merged[-1][0] == leaf_id:
            prev_leaf, prev_segment = merged[-1]
            merged[-1] = (prev_leaf, Interval(prev_segment.low, segment.high))
        else:
            merged.append((leaf_id, segment))
    return merged


def extract_tree_paths(
    oracle: DecisionTreeOracle,
    n_features: int,
    bounds: tuple[float, float] = (-1.0, 1.0),
    epsilon: float = 1e-4,
) -> TreeExtractionResult:
    """Extract leaf paths with bottom-up differential path finding."""
    root_intervals = tuple(Interval(bounds[0], bounds[1]) for _ in range(n_features))
    pending: list[np.ndarray] = [_representative(root_intervals)]
    seen_queries: set[tuple[float, ...]] = set()
    leaves: dict[int, ExtractedLeaf] = {}

    while pending:
        query = pending.pop()
        key = tuple(round(float(value), 10) for value in query)
        if key in seen_queries:
            continue
        seen_queries.add(key)

        leaf_id, label = oracle.query_leaf_and_label(query)
        if leaf_id in leaves:
            continue

        current_intervals = list(root_intervals)
        for feature_idx in range(n_features):
            base = _representative(tuple(current_intervals))
            base[feature_idx] = query[feature_idx]
            segments = _discover_feature_segments(
                oracle=oracle,
                base=base,
                feature_idx=feature_idx,
                interval=current_intervals[feature_idx],
                epsilon=epsilon,
            )

            current_segment = None
            for segment_leaf, segment in segments:
                if segment_leaf == leaf_id and segment.contains(query[feature_idx], eps=epsilon):
                    current_segment = segment
                    continue

                next_query = np.array(query, copy=True)
                next_query[feature_idx] = (segment.low + segment.high) / 2.0
                pending.append(next_query)

            if current_segment is not None:
                current_intervals[feature_idx] = current_segment

        leaves[leaf_id] = ExtractedLeaf(
            leaf_id=leaf_id,
            prediction=label,
            intervals=tuple(current_intervals),
        )

    return TreeExtractionResult(
        leaves=tuple(sorted(leaves.values(), key=lambda leaf: leaf.leaf_id)),
        query_count=oracle.query_count,
    )


def evaluate_extracted_tree(
    oracle: DecisionTreeOracle,
    extracted: TreeExtractionResult,
    X_test: np.ndarray,
    rng: np.random.Generator,
    uniform_samples: int,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, float]:
    fallback = extracted.leaves[0].prediction if extracted.leaves else 0
    clone = ExtractedTree(extracted.leaves, fallback_label=fallback)
    X_uniform = sample_uniform(uniform_samples, X_test.shape[1], rng, bounds=bounds)

    y_test_true = oracle.query_label(X_test)
    y_test_clone = clone.predict(X_test)
    y_uniform_true = oracle.query_label(X_uniform)
    y_uniform_clone = clone.predict(X_uniform)

    return {
        "test_agreement": float(np.mean(y_test_true == y_test_clone)),
        "uniform_agreement": float(np.mean(y_uniform_true == y_uniform_clone)),
    }
