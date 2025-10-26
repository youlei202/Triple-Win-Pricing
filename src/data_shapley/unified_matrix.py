"""Utilities for building the unified Shapley value matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from loguru import logger

from .data_loader import DataLoader
from .shapley import SUPPORTED_MODELS, evaluate_data_shapley

DEFAULT_DATASETS = (
    "iris",
    "titanic",
    "citeseer",
    "cora",
    "breast_cancer",
    "digits",
    "wine",
)


class UnifiedShapleyMatrix:
    """Compute a matrix of Shapley values across datasets and models."""

    def __init__(
        self,
        total_providers: int = 10,
        sample_number: int = 60,
        datasets: Sequence[str] = DEFAULT_DATASETS,
        models: Sequence[str] = SUPPORTED_MODELS,
        loader: DataLoader | None = None,
    ) -> None:
        self.total_providers = total_providers
        self.sample_number = sample_number
        self.datasets = tuple(datasets)
        self.models = tuple(models)
        self.loader = loader or DataLoader()
        self.provider_ids = list(range(total_providers))

    def build(
        self,
        output_path: str | Path | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """Compute the matrix and optionally persist it to disk."""
        records: List[dict] = []

        for dataset in self.datasets:
            for model in self.models:
                logger.info(
                    "Evaluating (%s, %s) with %d samples",
                    dataset,
                    model,
                    self.sample_number,
                )
                accuracy, shapley_values = evaluate_data_shapley(
                    model,
                    dataset,
                    total_providers=self.total_providers,
                    sample_number=self.sample_number,
                    loader=self.loader,
                    provider_indices=self.provider_ids,
                )

                row = {
                    "dataset_model": f"{dataset}_{model}",
                    "model_accuracy": accuracy,
                    "shapley_sum": sum(shapley_values.values()),
                }
                row.update(
                    {
                        f"seller_{provider_id}": shapley_values.get(provider_id, float("nan"))
                        for provider_id in self.provider_ids
                    }
                )
                records.append(row)

        df = pd.DataFrame(records).set_index("dataset_model")

        if save:
            path = (
                Path(output_path)
                if output_path is not None
                else Path("tables")
                / f"unified_shapley_matrix_{self.total_providers}sellers.csv"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
            logger.success("Unified Shapley matrix saved to %s", path)

        return df


__all__ = ["DEFAULT_DATASETS", "UnifiedShapleyMatrix"]

