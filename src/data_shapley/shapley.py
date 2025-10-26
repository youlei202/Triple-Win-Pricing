"""Core Shapley value evaluation logic."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data_loader import DataLoader


def _build_lr():
    return LogisticRegression(max_iter=1000, random_state=42)


def _build_svm():
    return SVC(decision_function_shape="ovo", probability=False, random_state=42)


def _build_rf():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def _build_dt():
    return DecisionTreeClassifier(random_state=42)


def _build_gb():
    return GradientBoostingClassifier(random_state=42)


def _build_knn():
    return KNeighborsClassifier(n_neighbors=5)


def _build_mlp():
    return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)


MODEL_BUILDERS = {
    "lr": _build_lr,
    "svm": _build_svm,
    "rf": _build_rf,
    "dt": _build_dt,
    "gb": _build_gb,
    "knn": _build_knn,
    "mlp": _build_mlp,
}

SUPPORTED_MODELS = tuple(MODEL_BUILDERS.keys())


def _score_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train and score the requested model, guarding against degenerate inputs."""
    if len(y_train) == 0:
        return 0.0

    if np.unique(y_train).shape[0] < 2:
        return 0.0

    try:
        estimator = MODEL_BUILDERS[model_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported model '{model_name}'.") from exc

    try:
        estimator.fit(X_train, y_train)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Failed to fit model '%s' on %d samples: %s",
            model_name,
            len(y_train),
            exc,
        )
        return 0.0

    predictions = estimator.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def evaluate_data_shapley(
    model_name: str,
    dataset: str,
    total_providers: int,
    sample_number: int,
    loader: DataLoader | None = None,
    provider_indices: Sequence[int] | None = None,
    random_seed: int | None = None,
) -> Tuple[float, Dict[int, float]]:
    """Estimate data Shapley values for the specified model and dataset."""
    loader = loader or DataLoader()
    provider_indices = (
        list(provider_indices)
        if provider_indices is not None
        else list(range(total_providers))
    )

    if not provider_indices:
        raise ValueError("At least one provider is required for Shapley evaluation.")

    bundle = loader.load_dataset(dataset, total_providers)
    full_X, full_y = bundle.combine(provider_indices)

    accuracy = _score_model(
        model_name, full_X, full_y, bundle.test_features, bundle.test_labels
    )

    rng = random.Random(random_seed)
    shapley_values = {provider_id: 0.0 for provider_id in provider_indices}

    for _ in range(sample_number):
        permutation = provider_indices[:]
        rng.shuffle(permutation)

        cumulative: List[int] = []
        previous_accuracy = 0.0

        for provider_id in permutation:
            cumulative.append(provider_id)
            subset_X, subset_y = bundle.combine(cumulative)
            current_accuracy = _score_model(
                model_name, subset_X, subset_y, bundle.test_features, bundle.test_labels
            )
            shapley_values[provider_id] += current_accuracy - previous_accuracy
            previous_accuracy = current_accuracy

    for provider_id in shapley_values:
        shapley_values[provider_id] /= sample_number

    total = sum(shapley_values.values())
    if abs(total) > 1e-12:
        shapley_values = {
            provider_id: value / total for provider_id, value in shapley_values.items()
        }
    else:
        shapley_values = {provider_id: 0.0 for provider_id in shapley_values}

    return accuracy, shapley_values


__all__ = ["SUPPORTED_MODELS", "evaluate_data_shapley"]
