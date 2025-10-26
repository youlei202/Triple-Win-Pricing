# Triple-Win Pricing & Data Shapley Toolkit

This repository contains two complementary building blocks for studying data-market dynamics:

- **Pricing engine (`src/pricing`)** – core algorithms for allocating prices between data providers, model brokers, and end buyers in a multi-sided “triple-win” marketplace.
- **Data Shapley pipeline (`src/data_shapley`)** – utilities to quantify each provider’s marginal contribution to downstream model accuracy across a catalog of datasets and learning algorithms. Results are stored in `tables/unified_shapley_matrix_10sellers.csv`.

Together they support the experiments documented in the notebooks under `notebooks/`, which explore fairness, deal success rates, feasible regions, commission fees, and algorithmic convergence.

---

## Repository Layout

```
notebooks/                       Exploratory analyses and experiment walkthroughs
  calculate_data_shapley.ipynb   Regenerates the unified Shapley value matrix CSV
  experiment_*.ipynb             Pricing experiments (fairness, success rate, etc.)
plots/                           Saved figures used in the manuscripts/notebooks
src/
  data_shapley/                  Production-ready Shapley computation module
    data_loader.py               Dataset preparation and provider partitioning
    shapley.py                   Model registry and sampling-based Shapley estimator
    unified_matrix.py            Batch utility to build the unified CSV
  pricing/                       Triple-win pricing solvers and buyer/seller models
  Shapley.py                     Legacy shim importing from `data_shapley`
tables/                          Derived artifacts (currently the unified CSV)
requirements.txt                 Minimal runtime dependencies
```

---

## Installation

1. Create and activate a Python 3.9+ environment.
2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. If you plan to run the notebooks, install an IPython kernel (optional but recommended):
   ```bash
   pip install ipykernel jupyterlab
   ```

---

## Data Requirements

The Shapley pipeline can automatically fetch most datasets through scikit-learn, but a few large resources must be supplied manually under `src/data_shapley/data/`:

| Dataset       | Expectation (relative to `src/data_shapley/data/`)                       |
| ------------- | ------------------------------------------------------------------------ |
| `titanic`     | `titanic.csv` with columns: survived, pclass, age, sibsp, parch, fare    |
| `citeseer`    | `CiteSeer/raw/ind.citeseer.allx` and `ind.citeseer.ally` (Planetoid fmt) |
| `cora`        | `Cora/raw/ind.cora.allx` and `ind.cora.ally`                             |

All other datasets (iris, breast_cancer, digits, wine) are loaded via `sklearn.datasets`. Large text/network corpora such as 20 Newsgroups and KDD Cup 99 are currently disabled; add the required files and extend `DEFAULT_DATASETS` if you need them.

---

## Reproducing the Unified Shapley Matrix

1. Confirm the required data files exist (see above).
2. Launch Jupyter and open `notebooks/calculate_data_shapley.ipynb`.
3. Run the notebook. It will:
   - Inject `src/` into `sys.path`.
   - Instantiate `data_shapley.UnifiedShapleyMatrix` with 10 providers and 60 samples.
   - Save the matrix to `tables/unified_shapley_matrix_10sellers.csv`.

The CSV header lists sellers (`seller_0` … `seller_9`), the observed accuracy for each dataset–model combination, and a sanity-check sum of Shapley values (should equal 1.0).

---

## Working with the Pricing Engine

The pricing solvers operate on:

- A matrix of Shapley allocations (`SV_{i,j}`) linking datasets `i` to models `j`.
- Buyer blocks (see `pricing/buyer.py`) specifying demand curves, reservation prices, and learning rates.
- Seller-side bounds and initialization values.

Key entry points:

- `pricing.triple_win.TripleWinPricing` – full triple-win iteration with convergence checks.
- `pricing.broker_centric.BrokerCentricPricing` – variant emphasizing broker objectives.
- `pricing.average.AveragePricing` – baseline update strategy for comparison.

Each solver exposes iterative routines that update dataset-to-model payments (`p_DtoM`) and model-to-buyer prices (`p_MtoB`) until the loss metric defined in `pricing.base._PricingBase` falls below tolerance.

Refer to the experiment notebooks for concrete parameterizations and visualization code. The notebooks expect Shapley values in the format produced by `data_shapley`.

---

## Development Notes

- The legacy `src/Shapley.py` module now forwards to `data_shapley.evaluate_data_shapley` for backward compatibility. New code should import directly from `data_shapley`.
- `src/data_shapley/unified_matrix.py` chooses 60 Monte Carlo samples by default; adjust `sample_number` in the notebook or when instantiating `UnifiedShapleyMatrix` if you need faster (but noisier) runs.
- Long-running experiments can be scripted using the same APIs shown in the notebooks; the modular design keeps data loading, Shapley estimation, and pricing logic separate.

---

## License

This project is distributed under the terms of the LICENSE file included in the repository.

