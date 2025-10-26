"""Dataset loading utilities used for data Shapley experiments."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split


@dataclass
class DatasetBundle:
    """Container storing provider level splits and the shared evaluation set."""

    train_features: List[np.ndarray]
    train_labels: List[np.ndarray]
    test_features: np.ndarray
    test_labels: np.ndarray

    def combine(self, provider_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return training data for the selected providers."""
        if not provider_indices:
            feature_dim = (
                self.train_features[0].shape[1] if self.train_features else 0
            )
            return (
                np.empty((0, feature_dim), dtype=np.float64),
                np.empty((0,), dtype=self.test_labels.dtype),
            )

        features = [self.train_features[i] for i in provider_indices]
        labels = [self.train_labels[i] for i in provider_indices]
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


class DataLoader:
    """Load and split datasets so each provider owns a disjoint partition."""

    GRAPH_RAW_DIRS = {"citeseer": "CiteSeer", "cora": "Cora"}

    def __init__(
        self,
        data_root: Path | str | None = None,
        random_state: int = 42,
        test_size: float = 0.3,
    ) -> None:
        self.data_root = (
            Path(data_root)
            if data_root is not None
            else Path(__file__).resolve().parent / "data"
        )
        self.random_state = random_state
        self.test_size = test_size
        self._cache: Dict[Tuple[str, int], DatasetBundle] = {}

    def load_dataset(self, name: str, total_providers: int) -> DatasetBundle:
        """Return a cached dataset bundle prepared for the requested providers."""
        dataset_key = (name.lower(), total_providers)
        if dataset_key not in self._cache:
            loader = getattr(self, f"_load_{dataset_key[0]}", None)
            if loader is None:
                raise ValueError(f"Unsupported dataset '{name}'.")
            logger.debug(
                "Loading dataset '%s' for %d providers", name, total_providers
            )
            self._cache[dataset_key] = loader(total_providers)
        return self._cache[dataset_key]

    # ------------------------------------------------------------------ #
    # Dataset specific loaders                                           #
    # ------------------------------------------------------------------ #

    def _load_iris(self, total_providers: int) -> DatasetBundle:
        features, labels = load_iris(return_X_y=True)
        return self._create_bundle(features, labels, total_providers, stratify=labels)

    def _load_breast_cancer(self, total_providers: int) -> DatasetBundle:
        features, labels = load_breast_cancer(return_X_y=True)
        return self._create_bundle(features, labels, total_providers, stratify=labels)

    def _load_digits(self, total_providers: int) -> DatasetBundle:
        features, labels = load_digits(return_X_y=True)
        return self._create_bundle(features, labels, total_providers, stratify=labels)

    def _load_wine(self, total_providers: int) -> DatasetBundle:
        features, labels = load_wine(return_X_y=True)
        return self._create_bundle(features, labels, total_providers, stratify=labels)

    def _load_titanic(self, total_providers: int) -> DatasetBundle:
        path = self.data_root / "titanic.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected Titanic dataset at '{path}'. Provide the CSV file before "
                "computing Shapley values."
            )

        df = pd.read_csv(path)
        df.columns = [col.lower() for col in df.columns]

        required_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                "Titanic dataset is missing required columns: "
                + ", ".join(missing)
            )

        features = df[["pclass", "age", "sibsp", "parch", "fare"]].apply(
            pd.to_numeric, errors="coerce"
        )
        labels = pd.to_numeric(df["survived"], errors="coerce")

        valid = features.dropna().index.intersection(labels.dropna().index)
        features = features.loc[valid].astype(np.float64)
        labels = labels.loc[valid].astype(np.int64)

        return self._create_bundle(
            features.to_numpy(np.float64),
            labels.to_numpy(np.int64),
            total_providers,
            stratify=labels.to_numpy(np.int64),
        )

    def _load_citeseer(self, total_providers: int) -> DatasetBundle:
        return self._load_graph_dataset("citeseer", total_providers)

    def _load_cora(self, total_providers: int) -> DatasetBundle:
        return self._load_graph_dataset("cora", total_providers)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _create_bundle(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        total_providers: int,
        stratify: np.ndarray | None = None,
    ) -> DatasetBundle:
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)

        if stratify is not None:
            stratify = np.asarray(stratify)
            unique = np.unique(stratify)
            if unique.shape[0] < 2:
                stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        feature_parts = [
            np.asarray(part, dtype=np.float64) for part in np.array_split(X_train, total_providers)
        ]
        label_parts = [
            np.asarray(part) for part in np.array_split(y_train, total_providers)
        ]

        return DatasetBundle(
            train_features=feature_parts,
            train_labels=label_parts,
            test_features=np.asarray(X_test, dtype=np.float64),
            test_labels=np.asarray(y_test),
        )

    def _load_graph_dataset(
        self, dataset_name: str, total_providers: int
    ) -> DatasetBundle:
        folder = self.GRAPH_RAW_DIRS.get(dataset_name)
        if folder is None:
            raise ValueError(f"Unsupported graph dataset '{dataset_name}'.")

        raw_dir = self.data_root / folder / "raw"
        required_files = [
            f"ind.{dataset_name}.allx",
            f"ind.{dataset_name}.ally",
        ]

        missing_files = [file for file in required_files if not (raw_dir / file).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing files for dataset '{dataset_name}' in '{raw_dir}': "
                + ", ".join(missing_files)
            )

        features = self._read_pickle(raw_dir / f"ind.{dataset_name}.allx")
        labels_one_hot = self._read_pickle(raw_dir / f"ind.{dataset_name}.ally")

        if hasattr(features, "toarray"):
            features = features.toarray()
        if hasattr(labels_one_hot, "toarray"):
            labels_one_hot = labels_one_hot.toarray()

        labels = np.argmax(labels_one_hot, axis=1)
        return self._create_bundle(features, labels, total_providers, stratify=labels)

    @staticmethod
    def _read_pickle(path: Path):
        with path.open("rb") as handle:
            try:
                return pickle.load(handle, encoding="latin1")
            except TypeError:
                return pickle.load(handle)


__all__ = ["DataLoader", "DatasetBundle"]
