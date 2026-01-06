"""pricing/cmab_hs.py

CMAB-HS baseline (Combinatorial Multi-Armed Bandit + Hierarchical Stackelberg).

This module implements (a lightly adapted version of) the mechanism described in:

  * "Crowdsensing Data Trading based on Combinatorial Multi-Armed Bandit and
     Stackelberg Game".

Paper summary (the parts we implement)
-------------------------------------
The paper considers an online crowdsensing data trading (CDT) system with:
  - sellers i in M: provide sensing data with unknown quality;
  - a platform: selects K sellers each round and posts a unit collection price p^t;
  - a consumer: pays a unit service price p_{J,t}.

The mechanism (Algorithm 1 in the paper) repeats for t=1..N rounds:
  1) Select sellers using a CMAB policy with UCB exploration.
  2) Given the selected sellers' *estimated* qualities, run a 3-stage
     Hierarchical Stackelberg (HS) game and use the closed-form SE:
        - seller effort (sensing time) tau_i^* (Eq. 20)
        - platform price p^* (Eq. 21)
        - consumer price p_J^* (Eq. 22)
  3) Observe sensing quality samples and update UCB statistics (Eqs. 17-19).

Mapping to this repository
--------------------------
This repository's pricing solvers output:
  - p_DtoM : (I, J) dataset->model prices
  - p_MtoB : list of length J, each (K_j,) model->buyer prices

To make CMAB-HS usable as a baseline here, we map:
  - sellers  <-> datasets D_i (i=1..I)
  - platform <-> the model producer for each model M_j
  - consumer <-> an aggregated representative buyer-side for M_j

For each model j, we run an independent CMAB-HS process over the I datasets.
We treat SV_{i|j} (Shapley value of dataset i for model j) as the seller's
"true" quality (unknown to the algorithm), and optionally add noise to mimic
quality learning.

Finally, we convert the HS equilibrium (unit) prices and sensing times to this
repo's edge prices as:
  - If dataset i is selected for model j in the final round:
        p_DtoM[i,j] = kappa_D[i] + p_j^* * tau_{i,j}^*
    else:
        p_DtoM[i,j] = kappa_D[i]

  - For each buyer k of model j (we use a common base price plus buyer offset):
        p_MtoB[j][k] = kappa_mb[j][k] + pJ_j^* * sum_i tau_{i,j}^*

Then we optionally clip:
  - buyer prices to reserves R
  - data prices to [kappa_D, bar_p_DtoM]

Notes / caveats
---------------
* This is an *adaptation* of the paper's CDT model into the repo's
  dataset->model->buyer graph. It is intended as a reasonably faithful baseline
  rather than a perfect one-to-one reproduction of the paper's experimental
  environment.
* If you want a more literal reproduction (e.g., PoIs L, regret curves vs. N),
  you should run CMAB-HS in its native CDT setting. Here we focus on producing
  a single set of prices compatible with the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pricing.base import _PricingBase


@dataclass(frozen=True)
class _HSResult:
    """Closed-form Stackelberg equilibrium outcome for one round."""

    pJ: float
    p: float
    tau: np.ndarray  # shape (K_selected,)
    A: float
    B: float
    q_mean: float
    Theta: float
    Lambda: float


def _clip_scalar(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _hs_equilibrium_closed_form(
    q_bar_sel: np.ndarray,
    a_sel: np.ndarray,
    b_sel: np.ndarray,
    *,
    theta: float,
    lam: float,
    omega_val: float,
    pJ_bounds: Tuple[float, float],
    p_bounds: Tuple[float, float],
    eps: float = 1e-12,
) -> _HSResult:
    """Compute the HS Stackelberg equilibrium using Eqs. (20)-(22).

    Parameters
    ----------
    q_bar_sel : (K,) array
        Estimated qualities of the selected sellers.
    a_sel, b_sel : (K,) arrays
        Seller cost parameters in the quadratic cost.
    theta, lam : float
        Platform aggregation cost parameters.
    omega_val : float
        Consumer valuation parameter (paper's ω > 1).
    pJ_bounds : (pJ_min, pJ_max)
        Feasible range of consumer unit price.
    p_bounds : (p_min, p_max)
        Feasible range of platform unit price.

    Returns
    -------
    _HSResult
        pJ^*, p^*, tau^* plus intermediate scalars.
    """

    # Safeguard: qualities must be positive for the paper's closed-form.
    q = np.maximum(q_bar_sel.astype(float), eps)
    a = np.maximum(a_sel.astype(float), eps)
    b = np.maximum(b_sel.astype(float), 0.0)

    # A and B are defined in Theorem 15/16.
    A = float(np.sum(1.0 / (2.0 * q * a)))
    B = float(np.sum(b / (2.0 * a)))

    # Mean quality of selected sellers (used in valuation).
    q_mean = float(np.mean(q))

    # Scalars in Theorem 16.
    denom = 2.0 * (1.0 + theta * A)
    Theta = float(A / denom)
    Lambda = float((lam * A - 2.0 * theta * B * A + B) / denom + B)

    # Eq. (22) / Eq. (33): consumer's optimal unit service price pJ^*.
    # Delta in Eq. (28): (q_mean*Lambda - 2)^2 + 8*Theta*omega*(q_mean^2)
    Delta = float((q_mean * Lambda - 2.0) ** 2 + 8.0 * Theta * omega_val * (q_mean**2))
    Delta = max(Delta, 0.0)

    pJ = float((3.0 * q_mean * Lambda + np.sqrt(Delta) - 2.0) / (4.0 * q_mean * max(Theta, eps)))
    pJ = _clip_scalar(pJ, pJ_bounds[0], pJ_bounds[1])

    # Eq. (21): platform's optimal unit data collection price p^*.
    p = float((pJ * A - (lam * A - 2.0 * theta * B * A + B)) / (2.0 * A * (1.0 + theta * A)))
    p = _clip_scalar(p, p_bounds[0], p_bounds[1])

    # Eq. (20): each seller's optimal sensing time tau_i^*.
    tau = (p - q * b) / (2.0 * q * a)
    tau = np.maximum(tau, 0.0)

    return _HSResult(pJ=pJ, p=p, tau=tau, A=A, B=B, q_mean=q_mean, Theta=Theta, Lambda=Lambda)


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    n: int,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Sample from a clipped normal distribution."""

    if std <= 0.0:
        return np.full(n, mean, dtype=float)
    x = rng.normal(loc=mean, scale=std, size=n)
    return np.clip(x, lo, hi).astype(float)


class CMABHSPricing(_PricingBase):
    """CMAB-HS baseline.

    Parameters (beyond _PricingBase)
    --------------------------------
    N_rounds : int
        Total online rounds (paper's N).
    K_select : int or Sequence[int]
        Number of sellers selected each round (paper's K). If a sequence is
        provided, it must have length J and gives K per model.
    L : int
        Number of quality samples observed per selected seller per round
        (paper's L, i.e., number of PoIs).
    tau0 : float
        Initial sensing time used in the exploration round t=1.
    theta, lam : float
        Platform aggregation cost parameters (paper's θ, λ).
    omega_val : float
        Consumer valuation parameter in ϕ(·) = ω ln(1 + q_mean * total_tau).
        Must be > 1 in the paper.
    a, b : None | float | (I,) array
        Seller cost parameters in Ci = (a_i tau^2 + b_i tau) q_i.
        If b is None, defaults to C_var.
        If a is None, defaults to a_scale (see below).
    a_scale : Optional[float]
        Used only when a is None; default is 2*I (empirically produces sensible
        tau magnitudes when q is around 1/I).
    quality_noise_std : float
        Std. dev. for quality observations around the true quality SV_{i|j}.
        Set to 0.0 for deterministic learning.
    seed : int
        RNG seed.
    clip_buyer_to_R : bool
        Clip p_MtoB to reserves R.
    floor_data_to_kappa : bool
        Floor p_DtoM to kappa_D.
    clip_data_to_cap : bool
        Clip p_DtoM to bar_p_DtoM.
    """

    def __init__(
        self,
        *args,
        N_rounds: int = 50,
        K_select: Union[int, Sequence[int]] = 5,
        L: int = 1,
        tau0: float = 0.1,
        theta: float = 0.1,
        lam: float = 0.1,
        omega_val: float = 20.0,
        a: Optional[Union[float, np.ndarray]] = None,
        b: Optional[Union[float, np.ndarray]] = None,
        a_scale: Optional[float] = None,
        quality_noise_std: float = 0.0,
        seed: int = 0,
        clip_buyer_to_R: bool = True,
        floor_data_to_kappa: bool = True,
        clip_data_to_cap: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.N_rounds = int(N_rounds)
        if self.N_rounds < 1:
            raise ValueError("N_rounds must be >= 1")

        self.K_select = K_select
        self.L = int(L)
        if self.L < 1:
            raise ValueError("L must be >= 1")

        self.tau0 = float(tau0)
        if self.tau0 <= 0.0:
            raise ValueError("tau0 must be > 0")

        self.theta = float(theta)
        self.lam = float(lam)
        self.omega_val = float(omega_val)
        if self.omega_val <= 1.0:
            raise ValueError("omega_val must be > 1 (as assumed in the paper)")

        self.quality_noise_std = float(quality_noise_std)
        if self.quality_noise_std < 0.0:
            raise ValueError("quality_noise_std must be >= 0")

        self.seed = int(seed)

        # Cost params for sellers.
        if a_scale is None:
            a_scale = 2.0 * float(self.I)
        self._a_scale = float(a_scale)

        self._a_param = a
        self._b_param = b

        self.clip_buyer_to_R = bool(clip_buyer_to_R)
        self.floor_data_to_kappa = bool(floor_data_to_kappa)
        self.clip_data_to_cap = bool(clip_data_to_cap)

        # Filled in during fit
        self.selection_: Optional[np.ndarray] = None
        self.tau_: Optional[np.ndarray] = None
        self.p_unit_: Optional[np.ndarray] = None
        self.pJ_unit_: Optional[np.ndarray] = None

    def _K_for_model(self, j: int) -> int:
        if isinstance(self.K_select, (list, tuple, np.ndarray)):
            if len(self.K_select) != self.J:
                raise ValueError("If K_select is a sequence, it must have length J")
            K = int(self.K_select[j])
        else:
            K = int(self.K_select)
        return max(1, min(K, self.I))

    def _seller_cost_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve seller cost parameters a_i, b_i as arrays of shape (I,)."""

        # a
        if self._a_param is None:
            a_arr = np.full(self.I, self._a_scale, dtype=float)
        else:
            a_arr = np.asarray(self._a_param, dtype=float)
            if a_arr.ndim == 0:
                a_arr = np.full(self.I, float(a_arr), dtype=float)
            if a_arr.shape != (self.I,):
                raise ValueError("a must be scalar or shape (I,)")

        # b
        if self._b_param is None:
            b_arr = np.asarray(self.C_var, dtype=float).copy()
        else:
            b_arr = np.asarray(self._b_param, dtype=float)
            if b_arr.ndim == 0:
                b_arr = np.full(self.I, float(b_arr), dtype=float)
            if b_arr.shape != (self.I,):
                raise ValueError("b must be scalar or shape (I,)")

        # Enforce constraints (paper): a_i > 0, b_i >= 0.
        a_arr = np.maximum(a_arr, 1e-12)
        b_arr = np.maximum(b_arr, 0.0)
        return a_arr, b_arr

    def _initial_pJ_min_nonneg_profit(
        self,
        *,
        p: float,
        total_tau: float,
        pJ_bounds: Tuple[float, float],
    ) -> float:
        """Algorithm 1 step 4: pJ = arg min_{pJ} s.t. Omega >= 0.

        Platform profit (paper Eq. 7-8):
            Omega = pJ*T - p*T - (theta*T^2 + lam*T)

        Omega >= 0  <=>  pJ >= p + lam + theta*T.
        """

        pJ_req = float(p + self.lam + self.theta * total_tau)
        return _clip_scalar(pJ_req, pJ_bounds[0], pJ_bounds[1])

    def _run_cmab_hs_one_model(
        self,
        j: int,
        *,
        rng: np.random.Generator,
        a_all: np.ndarray,
        b_all: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """Run CMAB-HS for a single model j.

        Returns
        -------
        chi_final : (I,) bool
        pJ_final : float
        p_final : float
        tau_final : (I,) float (0 if not selected)
        """

        M = self.I
        K = self._K_for_model(j)
        L = self.L

        # True (unknown) qualities for this model, from Shapley values.
        q_true = np.asarray(self.SV[:, j], dtype=float)
        # Ensure within [0,1] and avoid exact zeros.
        q_true = np.clip(q_true, 0.0, 1.0)

        # Bandit state.
        n = np.zeros(M, dtype=float)  # n_i^t
        q_bar = np.zeros(M, dtype=float)  # \bar{q}_i^t
        q_hat = np.zeros(M, dtype=float)  # \hat{q}_i^t (UCB)

        # Helper to update UCB after updating (n, q_bar)
        def update_ucb() -> None:
            total_n = float(np.sum(n))
            total_n = max(total_n, 1.0)  # log safety
            # Eq. (19): epsilon_i^t = sqrt( (K+1) ln(sum n) / n_i )
            eps_i = np.sqrt((K + 1.0) * np.log(total_n) / np.maximum(n, 1e-12))
            q_hat[:] = q_bar + eps_i

        # Bounds (paper inputs) mapped from this repo.
        blk = self.buyers[j]
        pJ_bounds = (0.0, float(np.min(blk.R)))
        p_bounds = (float(np.min(self.C_var)), float(np.max(self.bar_p_DtoM[:, j])))

        # Store last-round strategies.
        chi = np.zeros(M, dtype=bool)
        tau = np.zeros(M, dtype=float)
        pJ = 0.0
        p = 0.0

        # ---- Round t = 1 (exploration): select all sellers ----
        chi[:] = True
        tau[:] = self.tau0
        p = p_bounds[1]  # pmax

        total_tau = float(np.sum(tau))
        pJ = self._initial_pJ_min_nonneg_profit(p=p, total_tau=total_tau, pJ_bounds=pJ_bounds)

        # Observe qualities L times per seller; update n and q_bar.
        # Eq. (17)-(18)
        for i in range(M):
            samples = _truncated_normal(rng, mean=float(q_true[i]), std=self.quality_noise_std, n=L)
            n[i] += L
            q_bar[i] = float(np.sum(samples) / n[i])  # since previous n=0

        update_ucb()

        # ---- Rounds t = 2..N ----
        for _t in range(2, self.N_rounds + 1):
            # Select top-K by UCB.
            top_idx = np.argsort(-q_hat)[:K]
            chi[:] = False
            chi[top_idx] = True

            # HS game among selected sellers.
            q_sel = q_bar[top_idx]
            a_sel = a_all[top_idx]
            b_sel = b_all[top_idx]

            hs = _hs_equilibrium_closed_form(
                q_sel,
                a_sel,
                b_sel,
                theta=self.theta,
                lam=self.lam,
                omega_val=self.omega_val,
                pJ_bounds=pJ_bounds,
                p_bounds=p_bounds,
            )
            pJ = hs.pJ
            p = hs.p

            tau[:] = 0.0
            tau[top_idx] = hs.tau

            # Observe qualities for selected sellers and update q_bar/n.
            for i in top_idx:
                samples = _truncated_normal(rng, mean=float(q_true[i]), std=self.quality_noise_std, n=L)
                # Incremental mean update
                prev_n = n[i]
                n[i] = prev_n + L
                q_bar[i] = float((prev_n * q_bar[i] + float(np.sum(samples))) / n[i])

            update_ucb()

        return chi.copy(), float(pJ), float(p), tau.copy()

    def fit(self) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        a_all, b_all = self._seller_cost_params()

        selection = np.zeros((self.I, self.J), dtype=bool)
        tau_mat = np.zeros((self.I, self.J), dtype=float)
        p_unit = np.zeros(self.J, dtype=float)
        pJ_unit = np.zeros(self.J, dtype=float)

        # Run per-model CMAB-HS
        for j in range(self.J):
            # Use an independent RNG stream per model for reproducibility.
            rng_j = np.random.default_rng(rng.integers(0, 2**32 - 1))
            chi_j, pJ_j, p_j, tau_j = self._run_cmab_hs_one_model(j, rng=rng_j, a_all=a_all, b_all=b_all)

            selection[:, j] = chi_j
            tau_mat[:, j] = tau_j
            p_unit[j] = p_j
            pJ_unit[j] = pJ_j

            # Convert to repo's edge prices
            # Data side
            pD = self.kappa_D + p_j * tau_j
            if self.floor_data_to_kappa:
                pD = np.maximum(pD, self.kappa_D)
            if self.clip_data_to_cap:
                pD = np.minimum(pD, self.bar_p_DtoM[:, j])
            self.p_DtoM[:, j] = pD

            # Buyer side
            total_tau = float(np.sum(tau_j))
            base_price = pJ_j * total_tau
            blk = self.buyers[j]
            pM = blk.kappa_mb + base_price
            if self.clip_buyer_to_R:
                pM = np.minimum(pM, blk.R)
            self.p_MtoB[j] = pM.astype(float)

        # Save for external use
        self.selection_ = selection
        self.tau_ = tau_mat
        self.p_unit_ = p_unit
        self.pJ_unit_ = pJ_unit

        out = self.export()
        out.update(
            {
                "selection": selection,
                "tau": tau_mat,
                "p_unit": p_unit,
                "pJ_unit": pJ_unit,
                "rounds": int(self.N_rounds),
                "K_select": [self._K_for_model(j) for j in range(self.J)],
                "L": int(self.L),
            }
        )
        return out
