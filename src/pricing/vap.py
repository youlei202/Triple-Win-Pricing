from __future__ import annotations

"""pricing/vap.py

VAP baseline (Xu et al., "VAP: Online Data Valuation and Pricing for Machine Learning
Models in Mobile Health", IEEE TMC 2024 / INFOCOM 2022).

The original paper proposes:
  (i)  an online *valuation* metric based on Bayesian entropy reduction, and
  (ii) an online *posted-price* mechanism (VAP-Pricing) framed as a contextual
       bandit problem with monotonic feedback augmentation.

This repository's pricing framework, however, starts from a pre-computed
"contribution" matrix SV_{i|j} (Shapley-like) and focuses on coupled prices
  - p_{D_i→M_j}: dataset i paid by model j
  - p_{M_j→B_k}: buyer k pays for model j

So, to make VAP usable as a baseline *without changing the rest of the code*,
we implement an adaptation that:
  1) Treats SV_{i|j} (after optional non-negativity & column normalization) as
     a proxy for per-dataset "data valuation" under each model.
  2) Converts that valuation into a monetary value using buyers' willingness
     to pay (buyer reserves) as a revenue proxy.
  3) Runs a per-model version of Algorithm 1 (VAP-Pricing) to post a price to
     each dataset (one arrival per dataset), using acceptance feedback derived
     from dataset reserve values (kappa_D).
  4) Sets model→buyer prices via the downstream pass-through rule used in the
     paper's ecosystem (i.e., forward_quote()).

Important notes
---------------
* This is NOT a faithful re-implementation of the paper's entropy-based
  VAP-Valuation, because this codebase does not maintain Bayesian posteriors or
  feature vectors x. Instead, it focuses on the paper's *pricing mechanism*
  (contextual posted-price with monotonic feedback augmentation).
* By default, if a dataset rejects the posted price, we record p_{D→M}=0 for
  that edge (no trade). This will look "infeasible" under the repo's
  acceptance-region check (p_{D→M} >= kappa_D) but correctly reflects rejection.
  You can change this behavior via rejected_price.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Sequence

from pricing.base import _PricingBase


def _col_normalize_nonneg(SV: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ensure non-negativity and normalize each column to sum to 1.

    If a column becomes all-zeros after clipping, it is left as zeros.
    """
    SVp = np.maximum(SV, 0.0)
    col_sums = SVp.sum(axis=0, keepdims=True)
    out = np.zeros_like(SVp)
    mask = col_sums > eps
    out[:, mask[0]] = SVp[:, mask[0]] / col_sums[:, mask[0]]
    return out


def _make_price_grid(
    upper: float,
    K: int,
    lower: float = 0.0,
    include_lower: bool = True,
) -> np.ndarray:
    """Create an increasing price grid (arms) of length K."""
    upper = float(max(upper, lower))
    if K <= 0:
        raise ValueError("K must be positive")
    if K == 1:
        return np.array([upper], dtype=float)
    if include_lower:
        return np.linspace(lower, upper, K, dtype=float)
    # Exclude lower endpoint (mimic {i/K} in the paper)
    return np.linspace(lower, upper, K + 1, dtype=float)[1:]


class VAPPricing(_PricingBase):
    """VAP-Pricing baseline (Algorithm 1 style), adapted to this repo.

    Parameters (in addition to _PricingBase)
    ----------------------------------------
    K : int
        Number of candidate prices (arms) per model.
    alpha : float
        UCB exploration coefficient.
    revenue_scale : float
        Multiplier mapping (normalized) SV shares to monetary value.
        value_{ij} = revenue_scale * buyer_value_j * SV_share_{ij} - revenue_fee.
    revenue_fee : float
        Flat fee (epsilon) subtracted from each per-dataset value (can be 0).
    buyer_value_mode : {'weighted_reserve', 'min_reserve', 'mean_reserve'}
        How to convert a model's buyer block into a single revenue proxy.
    price_grid : optional sequence of floats
        If provided, uses the same grid for all models (will be sorted).
        If None, uses a per-model grid up to max(bar_p_DtoM[:, j]).
    ordering : {'desc_value', 'random', 'given'}
        Dataset arrival order for each model.
    seed : int | None
        RNG seed for ordering='random'.
    decay : float
        Optional diminishing-returns factor. If >0, effective value at step t is
            value_eff = value_base / (1 + decay * accepted_so_far).
    rejected_price : float
        What to store in p_{D→M} when a dataset rejects the offer.
        0.0 means "no trade"; kappa_D[i] is a common alternative.
    """

    def __init__(
        self,
        shapley_values: np.ndarray,
        delta: np.ndarray,
        kappa_D: np.ndarray,
        buyers: List[Any],
        C_var: np.ndarray,
        bar_p_DtoM: np.ndarray,
        p_DtoM_init: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        max_iter: int = 10000,
        verbose: bool = False,
        *,
        K: int = 10,
        alpha: float = 1.2,
        revenue_scale: float = 1.0,
        revenue_fee: float = 0.0,
        buyer_value_mode: str = "weighted_reserve",
        price_grid: Optional[Sequence[float]] = None,
        ordering: str = "desc_value",
        seed: Optional[int] = 0,
        decay: float = 0.0,
        rejected_price: float = 0.0,
        include_zero_price: bool = True,
        clip_and_normalize_sv: bool = True,
    ):
        super().__init__(
            shapley_values=shapley_values,
            delta=delta,
            kappa_D=kappa_D,
            buyers=buyers,
            C_var=C_var,
            bar_p_DtoM=bar_p_DtoM,
            p_DtoM_init=p_DtoM_init,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
        )

        if K <= 0:
            raise ValueError("K must be a positive integer")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if revenue_scale < 0:
            raise ValueError("revenue_scale must be non-negative")
        if decay < 0:
            raise ValueError("decay must be non-negative")

        self.K = int(K)
        self.alpha = float(alpha)
        self.revenue_scale = float(revenue_scale)
        self.revenue_fee = float(revenue_fee)
        self.buyer_value_mode = str(buyer_value_mode)
        self.ordering = str(ordering)
        self.seed = seed
        self.decay = float(decay)
        self.rejected_price = float(rejected_price)
        self.include_zero_price = bool(include_zero_price)
        self.clip_and_normalize_sv = bool(clip_and_normalize_sv)

        if price_grid is None:
            self.price_grid = None
        else:
            g = np.asarray(list(price_grid), dtype=float)
            if g.ndim != 1 or g.size == 0:
                raise ValueError("price_grid must be a 1D non-empty sequence")
            g = np.unique(np.sort(g))
            self.price_grid = g

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _buyer_value_proxy(self, j: int) -> float:
        """Convert buyer block j to a single monetary scale."""
        blk = self.buyers[j]
        mode = self.buyer_value_mode.lower()
        if mode in ("weighted_reserve", "weighted_r", "weighted"):
            return float(np.dot(blk.omega, blk.R))
        if mode in ("min_reserve", "min_r", "min"):
            return float(np.min(blk.R))
        if mode in ("mean_reserve", "avg_reserve", "mean_r", "avg", "mean"):
            return float(np.mean(blk.R))
        raise ValueError(
            "buyer_value_mode must be one of: 'weighted_reserve', 'min_reserve', 'mean_reserve'"
        )

    def _compute_value_matrix(self) -> np.ndarray:
        """Compute per-edge monetary values used as context features.

        Returns
        -------
        value : (I, J) ndarray
            Non-negative value_{ij} in the same currency scale as prices.
        """
        if self.clip_and_normalize_sv:
            shares = _col_normalize_nonneg(self.SV)
        else:
            # Use raw SV as-is (may contain negatives / non-normalized columns).
            shares = self.SV.copy().astype(float)

        buyer_val = np.array([self._buyer_value_proxy(j) for j in range(self.J)], dtype=float)

        value = self.revenue_scale * shares * buyer_val[None, :] - self.revenue_fee
        return np.maximum(value, 0.0)

    def _dataset_order(self, values_j: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return an ordering of dataset indices for one model."""
        ord_mode = self.ordering.lower()
        if ord_mode in ("given", "as_is", "asis", "natural"):
            return np.arange(self.I, dtype=int)
        if ord_mode in ("desc_value", "desc", "descending"):
            # High value first
            return np.argsort(-values_j)
        if ord_mode in ("random", "shuffle"):
            idx = np.arange(self.I, dtype=int)
            rng.shuffle(idx)
            return idx
        raise ValueError("ordering must be one of: 'desc_value', 'random', 'given'")

    # ---------------------------
    # Main API
    # ---------------------------
    def fit(self) -> Dict[str, Any]:
        """Run VAP-Pricing per model and then set model→buyer prices downstream."""
        rng = np.random.default_rng(self.seed)

        value_mat = self._compute_value_matrix()  # (I, J)

        # Reset p_DtoM
        self.p_DtoM = np.zeros((self.I, self.J), dtype=float)

        # Track some diagnostics
        accepted = np.zeros((self.I, self.J), dtype=bool)
        total_profit = 0.0

        # Process each model separately
        for j in range(self.J):
            values_j_base = value_mat[:, j].astype(float)

            # Candidate price grid (arms)
            if self.price_grid is None:
                upper = float(np.max(self.bar_p_DtoM[:, j]))
                if not np.isfinite(upper) or upper <= 0:
                    # Fall back to a conservative upper bound if caps are missing
                    upper = max(1.0, float(np.max(self.kappa_D)) * 3.0)
                p_grid = _make_price_grid(
                    upper=upper,
                    K=self.K,
                    lower=0.0,
                    include_lower=self.include_zero_price,
                )
            else:
                p_grid = self.price_grid
                # If user-supplied grid doesn't match K, we accept it as-is.
                if p_grid.size < 2:
                    # Degenerate grid: keep but safe expansion
                    p_grid = _make_price_grid(
                        upper=float(p_grid[0]),
                        K=max(self.K, 2),
                        lower=0.0,
                        include_lower=self.include_zero_price,
                    )

            K_eff = int(p_grid.size)

            # Ridge regression state for each arm (Algorithm 1): A_i (2x2), b_i (2,)
            A = np.repeat(np.eye(2, dtype=float)[None, :, :], K_eff, axis=0)  # (K,2,2)
            b = np.zeros((K_eff, 2), dtype=float)

            idx_order = self._dataset_order(values_j_base, rng)
            accepted_so_far = 0

            for t_idx in idx_order:
                v_base = float(values_j_base[t_idx])
                if self.decay > 0:
                    v_eff = v_base / (1.0 + self.decay * accepted_so_far)
                else:
                    v_eff = v_base

                # Context feature Π_t = (π(G), n)^T in the paper; here use (value, 1)
                Pi = np.array([v_eff, 1.0], dtype=float)

                # Compute UCB score for each arm
                ucb = np.empty(K_eff, dtype=float)
                for i_arm in range(K_eff):
                    theta = np.linalg.solve(A[i_arm], b[i_arm])
                    Ai_inv_Pi = np.linalg.solve(A[i_arm], Pi)
                    bonus = self.alpha * float(np.sqrt(max(0.0, Pi @ Ai_inv_Pi)))
                    ucb[i_arm] = float(Pi @ theta) + bonus

                It = int(np.argmax(ucb))

                # Posted price (paper line 8): p = min(p_{It}, value)
                posted = float(min(p_grid[It], v_eff))
                posted = max(0.0, posted)

                # Dataset reserve value (kappa_D) drives accept/reject
                reserve = float(self.kappa_D[t_idx])
                is_accept = posted >= reserve

                # Shared outer product term
                outer = np.outer(Pi, Pi)

                if is_accept:
                    accepted_so_far += 1

                    # Monotonic augmentation: if accepts p_{It}, then accepts any higher price
                    for i_arm in range(It, K_eff):
                        eff_price = float(min(p_grid[i_arm], v_eff))
                        reward = max(0.0, v_eff - eff_price)
                        A[i_arm] += outer
                        b[i_arm] += reward * Pi

                    self.p_DtoM[t_idx, j] = posted
                    accepted[t_idx, j] = True
                    total_profit += (v_eff - posted)
                else:
                    # If rejects p_{It}, then rejects any lower price
                    for i_arm in range(0, It + 1):
                        A[i_arm] += outer
                        # b unchanged because reward = 0

                    self.p_DtoM[t_idx, j] = self.rejected_price
                    accepted[t_idx, j] = False

        # After data pricing, set model→buyer prices by downstream pass-through.
        vM = self.forward_quote()
        for j in range(self.J):
            self.p_MtoB[j] = vM[j]

        out = self.export()
        out.update(
            {
                "total_profit_proxy": float(total_profit),
                "accepted_mask": accepted,
                "accepted_frac": float(np.mean(accepted)),
                "K": int(self.K),
                "alpha": float(self.alpha),
            }
        )
        return out
