"""Compatibility layer forwarding to ``data_shapley``."""

from __future__ import annotations

import warnings
from typing import Dict, Sequence, Tuple

from data_shapley import DataLoader, SUPPORTED_MODELS, evaluate_data_shapley


def eval_shapley(
    model: str,
    dataset: str,
    index: Sequence[int],
    total_provider: int,
    sample_number: int,
) -> Tuple[float, Dict[int, float]]:
    """Legacy wrapper around :func:`data_shapley.evaluate_data_shapley`."""
    warnings.warn(
        "Importing eval_shapley from 'Shapley' is deprecated. "
        "Use 'data_shapley.evaluate_data_shapley' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    provider_indices = list(index) if index else list(range(total_provider))
    loader = DataLoader()
    return evaluate_data_shapley(
        model_name=model,
        dataset=dataset,
        total_providers=total_provider,
        sample_number=sample_number,
        loader=loader,
        provider_indices=provider_indices,
    )


__all__ = ["eval_shapley", "SUPPORTED_MODELS"]

