"""
pricing/dealer.py

DealerPricing baseline
======================

This module implements a practical baseline inspired by:

  "Dealer: an end-to-end model marketplace with differential privacy"

The original Dealer framework proposes:
  1) Versioned models with different DP levels (epsilon),
  2) Arbitrage-free (relaxed) pricing across versions (DPP / DPP+),
  3) Budget-aware data procurement (subset selection) based on Shapley values,
  4) A compensation function c_i(epsilon) = b_i * exp(rho_i * epsilon).

This repository's simulator is more abstract: it does not train models or
simulate DP noise directly. Instead, it provides:
  - a Shapley matrix SV_{i|j} as a proxy for data contribution,
  - per-model buyer reserve prices R (willingness-to-pay),
  - seller cost floors kappa_D and buyer overhead kappa_mb.

To keep the baseline comparable and lightweight, we implement the key
mechanism components under these constraints:

  • Model-side pricing: use a DPP+-style dynamic program to choose one posted
    price per model, while enforcing relaxed arbitrage-free constraints across
    a quality axis epsilon (monotone price and non-increasing unit price).

  • Data procurement: allocate a fraction of predicted revenue as a
    "manufacturing budget" and select a subset of datasets under that budget
    (knapsack / greedy), maximizing Shapley value.

  • Data compensation: pay selected datasets proportionally to their
    compensation request c_i(epsilon), which is Shapley-linked but modulated
    by a privacy-sensitivity rho_i.

This is not a full reproduction of Dealer (no DP training loop, no buyer
requirement model). It is, however, a faithful implementation of the pricing
and procurement primitives that make Dealer distinct from simple markups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pricing.base import _PricingBase


@dataclass(frozen=True)
class DealerDiagnostics:
    """Extra metadata useful for debugging / reporting."""
    eps_grid: np.ndarray
    model_order: np.ndarray
    chosen_prices: np.ndarray
    predicted_sales: np.ndarray
    predicted_revenue: np.ndarray
    data_budgets: np.ndarray
    selected_counts: np.ndarray


class DealerPricing(_PricingBase):
    """Dealer-inspired baseline.

    Parameters
    ----------
    eps_grid:
        Optional array of shape (J,) giving epsilon per model.
        If None, a log-spaced grid is generated.
    eps_min, eps_max:
        Range used when generating eps_grid.
    sort_models:
        If True, internally sorts models by demand strength (median reserve)
        before running the DPP+ pricing DP. This aligns monotone constraints
        with the reserve structure.
    survey_frac:
        Fraction of buyers used as survey points per model (in (0,1]).
        The DP uses survey points to estimate demand; evaluation uses all buyers.
    broker_fee:
        Fraction of model revenue retained by the broker/platform (in [0,1)).
        The remaining (1-broker_fee) share is used as data procurement budget.
        Set to 0.0 to match the paper's neutral-broker assumption.
    rho_range:
        Tuple (rho_low, rho_high) controlling privacy-sensitivity rho_i.
        We map seller cost floors kappa_D monotonically into this range.
    b_scale:
        Scale factor for Shapley-based base prices b_{i|j}. If None, uses a
        robust scale derived from buyer reserves.
    exact_subset_threshold:
        If I (number of datasets) <= threshold, solve subset selection exactly
        by brute force. Otherwise, use a greedy ratio heuristic.
    seed:
        RNG seed for survey sampling.

    Notes
    -----
    Output conventions follow the repository's API:
      - p_MtoB[j] is an array of shape (K_j,) (here constant per buyer).
      - p_DtoM is an (I,J) matrix (here nonzero only for selected datasets).
    """

    def __init__(
        self,
        *args,
        eps_grid: Optional[np.ndarray] = None,
        eps_min: float = 0.2,
        eps_max: float = 5.0,
        sort_models: bool = True,
        survey_frac: float = 1.0,
        broker_fee: float = 0.0,
        rho_range: Tuple[float, float] = (0.10, 0.60),
        b_scale: Optional[float] = None,
        exact_subset_threshold: int = 18,
        seed: int | None = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if eps_min <= 0 or eps_max <= 0 or eps_max <= eps_min:
            raise ValueError("eps_min/eps_max must be positive and eps_max > eps_min")

        self.sort_models = bool(sort_models)

        if not (0.0 < survey_frac <= 1.0):
            raise ValueError("survey_frac must be in (0, 1]")
        self.survey_frac = float(survey_frac)

        if not (0.0 <= broker_fee < 1.0):
            raise ValueError("broker_fee must be in [0, 1)")
        self.broker_fee = float(broker_fee)

        rho_low, rho_high = float(rho_range[0]), float(rho_range[1])
        if rho_low < 0 or rho_high < rho_low:
            raise ValueError("rho_range must satisfy 0 <= rho_low <= rho_high")
        self.rho_low = rho_low
        self.rho_high = rho_high

        self.b_scale = b_scale
        self.exact_subset_threshold = int(exact_subset_threshold)

        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Epsilon grid (monotone increasing).
        if eps_grid is None:
            self.eps_grid = np.geomspace(eps_min, eps_max, num=self.J).astype(float)
        else:
            eg = np.asarray(eps_grid, dtype=float).reshape(-1)
            if eg.size != self.J:
                raise ValueError("eps_grid must have shape (J,)")
            if np.any(eg <= 0):
                raise ValueError("eps_grid must be strictly positive")
            self.eps_grid = np.sort(eg)

    # ---------------------------------------------------------------------
    # Pricing (DPP+ style DP under relaxed arbitrage-free constraints)
    # ---------------------------------------------------------------------
    @staticmethod
    def _unique_sorted(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0:
            return x
        x = np.unique(x)
        x = x[np.isfinite(x)]
        x = x[x >= 0.0]
        return np.sort(x)

    def _sample_survey(self, R: np.ndarray) -> np.ndarray:
        """Sample survey budgets from reserves."""
        R = np.asarray(R, dtype=float)
        if self.survey_frac >= 1.0 or R.size <= 1:
            return R.copy()
        m = max(1, int(np.ceil(self.survey_frac * R.size)))
        idx = self._rng.choice(R.size, size=m, replace=False)
        return R[idx].copy()

    def _build_complete_price_space(
        self,
        eps: np.ndarray,
        survey_budgets: List[np.ndarray],
        floors: np.ndarray,
    ) -> List[np.ndarray]:
        """Construct a complete price space (Algorithm-3 style).

        For each version k with epsilon eps[k], start from its survey budgets
        as candidate posted prices. Add:
          • SC points: scaled-up prices to higher-epsilon versions.
          • MC points: copied prices to lower-epsilon versions.

        This helps the DP not miss feasible optima due to cross-version constraints.
        """
        J = len(survey_budgets)
        cand_sets: List[set[float]] = []
        for k in range(J):
            base = self._unique_sorted(survey_budgets[k]).tolist()
            s = set(float(x) for x in base)
            s.add(float(floors[k]))  # ensure floor is feasible
            cand_sets.append(s)

        # Add SC / MC points.
        for k in range(J):
            eps_k = float(eps[k])
            for p in self._unique_sorted(survey_budgets[k]):
                p = float(p)
                if p <= 0:
                    continue

                # MC: add to all lower-epsilon versions.
                for k2 in range(0, k):
                    cand_sets[k2].add(p)

                # SC: scale to higher-epsilon versions.
                for k2 in range(k + 1, J):
                    cand_sets[k2].add(p * float(eps[k2]) / eps_k)

        cand_list: List[np.ndarray] = []
        for k in range(J):
            arr = np.array(sorted(cand_sets[k]), dtype=float)
            arr = arr[np.isfinite(arr) & (arr >= 0.0)]
            cand_list.append(arr)
        return cand_list

    @staticmethod
    def _revenue_curve(cand: np.ndarray, budgets: np.ndarray) -> np.ndarray:
        """Compute revenue(p) = p * #{budgets >= p} for each candidate p."""
        budgets = np.asarray(budgets, dtype=float)
        if budgets.size == 0:
            return np.zeros_like(cand)

        b_sorted = np.sort(budgets)
        n = b_sorted.size
        idx = np.searchsorted(b_sorted, cand, side="left")
        demand = (n - idx).astype(float)
        return cand * demand

    @staticmethod
    def _dp_opt_prices(
        eps: np.ndarray,
        cand_list: List[np.ndarray],
        rev_list: List[np.ndarray],
        tol: float = 1e-12,
    ) -> np.ndarray:
        """Dynamic program for relaxed arbitrage-free pricing.

        Constraints (k increases with eps):
          1) Monotone posted price:        p_k <= p_{k+1}
          2) Non-increasing unit price:    p_k/eps_k >= p_{k+1}/eps_{k+1}

        Returns
        -------
        p_star : (J,) ndarray
            Chosen candidate price for each version.
        """
        J = len(cand_list)
        if J == 0:
            return np.array([], dtype=float)

        dp: List[np.ndarray] = []
        back: List[np.ndarray] = []

        # Base case.
        dp0 = rev_list[0].copy()
        back0 = -np.ones_like(dp0, dtype=int)
        dp.append(dp0)
        back.append(back0)

        # Transitions.
        for k in range(1, J):
            cand_k = cand_list[k]
            cand_prev = cand_list[k - 1]
            eps_prev = float(eps[k - 1])
            eps_k = float(eps[k])

            dp_k = -np.inf * np.ones_like(cand_k)
            back_k = -np.ones_like(cand_k, dtype=int)

            # Brute-force transitions (small sizes in this repo).
            for j, p_k in enumerate(cand_k):
                best = -np.inf
                best_i = -1
                unit_k = p_k / eps_k

                for i, p_prev in enumerate(cand_prev):
                    if p_prev <= p_k + tol and (p_prev / eps_prev) + tol >= unit_k:
                        val = dp[k - 1][i]
                        if val > best:
                            best = val
                            best_i = i

                if best_i >= 0:
                    dp_k[j] = best + rev_list[k][j]
                    back_k[j] = best_i

            dp.append(dp_k)
            back.append(back_k)

        # Backtrack.
        p_star = np.zeros(J, dtype=float)
        j_star = int(np.nanargmax(dp[-1]))
        p_star[J - 1] = float(cand_list[J - 1][j_star])
        for k in range(J - 1, 0, -1):
            j_star = int(back[k][j_star])
            if j_star < 0:
                j_star = int(np.nanargmax(dp[k - 1]))
            p_star[k - 1] = float(cand_list[k - 1][j_star])
        return p_star

    # ---------------------------------------------------------------------
    # Data procurement (subset selection under budget)
    # ---------------------------------------------------------------------
    def _rho_from_kappaD(self) -> np.ndarray:
        kd = self.kappa_D
        if kd.size == 0:
            return kd.copy()
        lo, hi = float(np.min(kd)), float(np.max(kd))
        if hi - lo <= 1e-12:
            return np.full_like(kd, (self.rho_low + self.rho_high) / 2.0)
        t = (kd - lo) / (hi - lo)
        return self.rho_low + t * (self.rho_high - self.rho_low)

    @staticmethod
    def _exact_knapsack(values: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
        """Exact subset selection by brute force (2^I) for small I."""
        values = np.asarray(values, dtype=float)
        costs = np.asarray(costs, dtype=float)
        I = values.size
        best_val = -np.inf
        best_mask = 0

        # Enumerate all subsets. For I<=18 this is at most 262k.
        for mask in range(1 << I):
            total_cost = 0.0
            total_val = 0.0
            for i in range(I):
                if mask & (1 << i):
                    total_cost += float(costs[i])
                    if total_cost > budget:
                        break
                    total_val += float(values[i])
            else:
                if total_val > best_val:
                    best_val = total_val
                    best_mask = mask

        sel = [i for i in range(I) if (best_mask & (1 << i))]
        return np.array(sel, dtype=int)

    @staticmethod
    def _greedy_knapsack(values: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
        """Greedy ratio heuristic for knapsack."""
        values = np.asarray(values, dtype=float)
        costs = np.asarray(costs, dtype=float)
        ratio = values / (costs + 1e-12)
        order = np.argsort(ratio)[::-1]
        sel: List[int] = []
        spent = 0.0
        for i in order:
            c = float(costs[i])
            if spent + c <= budget + 1e-12:
                sel.append(int(i))
                spent += c
        return np.array(sel, dtype=int)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fit(self) -> Dict[str, Any]:
        I, J = self.I, self.J

        # (A) Internal model order for DPP+ constraints
        demand_strength = np.array([np.median(blk.R) for blk in self.buyers], dtype=float)
        if self.sort_models:
            order = np.argsort(demand_strength)  # low -> high demand
        else:
            order = np.arange(J)

        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(J)

        # Assign eps by rank (higher demand -> higher eps)
        eps_sorted = self.eps_grid.copy()
        eps_per_model = np.zeros(J, dtype=float)
        eps_per_model[order] = eps_sorted

        # Floors for posted prices (cover worst-case overhead in that buyer block)
        floors = np.array([float(np.max(self.buyers[j].kappa_mb)) for j in order], dtype=float)

        # (B) Survey budgets + complete price space
        survey_budgets_sorted: List[np.ndarray] = []
        for j in order:
            blk = self.buyers[j]
            survey_budgets_sorted.append(self._sample_survey(blk.R))

        cand_list = self._build_complete_price_space(
            eps=eps_sorted,
            survey_budgets=survey_budgets_sorted,
            floors=floors,
        )
        rev_list = [self._revenue_curve(cand_list[k], survey_budgets_sorted[k]) for k in range(J)]

        # (C) DPP+ DP to choose posted prices
        p_sorted = self._dp_opt_prices(eps=eps_sorted, cand_list=cand_list, rev_list=rev_list)

        # Map back to original model indices
        chosen_price = np.zeros(J, dtype=float)
        chosen_price[order] = p_sorted

        # Set buyer-side posted prices (constant per buyer)
        for j, blk in enumerate(self.buyers):
            self.p_MtoB[j] = np.full_like(blk.R, float(chosen_price[j]), dtype=float)

        # (D) Predict sales/revenue and allocate procurement budgets
        predicted_sales = np.zeros(J, dtype=float)
        predicted_revenue = np.zeros(J, dtype=float)
        data_budget = np.zeros(J, dtype=float)

        for j, blk in enumerate(self.buyers):
            p = float(chosen_price[j])
            sales = float(np.sum(np.asarray(blk.R, dtype=float) >= p))
            predicted_sales[j] = sales
            predicted_revenue[j] = p * sales
            data_budget[j] = (1.0 - self.broker_fee) * predicted_revenue[j]

        # (E) Data compensation + subset selection per model
        rho_vec = self._rho_from_kappaD()  # (I,)

        if self.b_scale is None:
            all_R = np.concatenate([np.asarray(blk.R, dtype=float) for blk in self.buyers])
            b_scale = float(np.median(all_R))
        else:
            b_scale = float(self.b_scale)

        self.p_DtoM = np.zeros((I, J), dtype=float)
        selected_counts = np.zeros(J, dtype=float)

        for j in range(J):
            budget = float(data_budget[j])
            if budget <= 1e-12:
                continue

            eps_j = float(eps_per_model[j])

            sv = np.asarray(self.SV[:, j], dtype=float)
            sv = np.clip(sv, 0.0, None)

            # Base price b_{i|j} is Shapley-linked and lower-bounded by kappa_D
            b_ij = self.kappa_D + b_scale * sv

            # Compensation request c_i(eps) = b_i * exp(rho_i * eps)
            c_ij = b_ij * np.exp(rho_vec * eps_j)

            # Subset selection: maximize sum(SV) subject to cost <= budget
            if I <= self.exact_subset_threshold:
                sel = self._exact_knapsack(values=sv, costs=c_ij, budget=budget)
            else:
                sel = self._greedy_knapsack(values=sv, costs=c_ij, budget=budget)

            if sel.size == 0:
                continue

            selected_counts[j] = float(sel.size)
            c_sel = c_ij[sel]
            sum_c = float(np.sum(c_sel))
            if sum_c <= 1e-12:
                continue

            # Pay selected datasets proportionally to their request.
            # Since sum_c <= budget by construction, scale >= 1 ensures IR.
            scale = max(1.0, budget / (sum_c + 1e-12))
            pay_sel = c_sel * scale

            self.p_DtoM[sel, j] = pay_sel

        out = self.export()
        out["diagnostics"] = DealerDiagnostics(
            eps_grid=eps_per_model.copy(),
            model_order=order.copy(),
            chosen_prices=chosen_price.copy(),
            predicted_sales=predicted_sales.copy(),
            predicted_revenue=predicted_revenue.copy(),
            data_budgets=data_budget.copy(),
            selected_counts=selected_counts.copy(),
        )
        return out
