from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pricing.buyer import BuyerBlock


class _PricingBase:
    """
    Base class for data–model coupled pricing (multi-buyer × multi-seller).

    This is a *no-clipping* base: prices are not projected to bounds at init
    or during iteration. The arrays C_var and bar_p_DtoM are kept for
    compatibility and may serve as convenient starting points only.

    Notation:
      I = number of datasets D_i
      J = number of models   M_j
      K_j = number of buyers for model j

    Parameters
    ----------
    shapley_values : (I, J) array_like
        SV_{i|j} ≥ 0 — contribution of dataset i to model j.
    delta : (J,) array_like
        Producer margins δ_j ≥ 0.
    kappa_D : (I,) array_like
        Dataset-side positive offsets κ_{D_i} > 0.
    buyers : list[BuyerBlock] of length J
        Per-model buyer blocks. Each block carries kappa_mb, p0, R, omega, p_init.
        NOTE: p0/R are not enforced as bounds in this base; p_init/p0 are used
        only as initial guesses.
    C_var : (I,) array_like
        Per-use variable-cost lower bound C_i^{var}. Used only as a default initializer.
    bar_p_DtoM : (I, J) array_like
        Upper caps \bar p_{i→M_j}. Kept for API compatibility; not enforced here.
    p_DtoM_init : (I, J) array_like, optional
        Optional initializer for p_{D→M}. If None, uses C_var[:, None].
    tol : float
        Convergence tolerance used by subclasses.
    max_iter : int
        Max iterations used by subclasses.
    verbose : bool
        Verbose flag used by subclasses.
    """

    def __init__(
        self,
        shapley_values: np.ndarray,
        delta: np.ndarray,
        kappa_D: np.ndarray,
        buyers: List[BuyerBlock],
        C_var: np.ndarray,
        bar_p_DtoM: np.ndarray,
        p_DtoM_init: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        max_iter: int = 10000,
        verbose: bool = False,
    ):
        # Store data (convert to float arrays)
        self.SV = np.asarray(shapley_values, dtype=float)
        self.delta = np.asarray(delta, dtype=float)
        self.kappa_D = np.asarray(kappa_D, dtype=float)
        self.buyers = buyers
        self.C_var = np.asarray(C_var, dtype=float)  # for initialization only
        self.bar_p_DtoM = np.asarray(
            bar_p_DtoM, dtype=float
        )  # kept for API compatibility
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = verbose

        # Shape checks
        self.I, self.J = self.SV.shape
        assert self.delta.shape == (self.J,), "delta must be shape (J,)"
        assert self.kappa_D.shape == (self.I,), "kappa_D must be shape (I,)"
        assert self.C_var.shape == (self.I,), "C_var must be shape (I,)"
        assert self.bar_p_DtoM.shape == (
            self.I,
            self.J,
        ), "bar_p_DtoM must be shape (I,J)"
        assert (
            len(self.buyers) == self.J
        ), "buyers list length must equal number of models J"

        # Buyer-side shapes and initialization (NO clipping)
        self.K_list = [blk.kappa_mb.size for blk in buyers]
        self.p_MtoB: List[np.ndarray] = []
        for j, blk in enumerate(buyers):
            for name, arr in [
                ("kappa_mb", blk.kappa_mb),
                ("p0", blk.p0),
                ("R", blk.R),
                ("omega", blk.omega),
            ]:
                assert arr.shape == (
                    self.K_list[j],
                ), f"{name} for model {j} must be shape (K_j,)"

            # Initial prices: use p_init if provided; otherwise use p0 as a numeric starting point.
            if blk.p_init is not None:
                p_start = np.asarray(blk.p_init, dtype=float)
            else:
                p_start = np.asarray(blk.p0, dtype=float)
            self.p_MtoB.append(p_start.copy())

        # Data→Model initialization (NO clipping)
        if p_DtoM_init is None:
            self.p_DtoM = np.repeat(self.C_var[:, None], self.J, axis=1).astype(float)
        else:
            P0 = np.asarray(p_DtoM_init, dtype=float)
            assert P0.shape == (self.I, self.J), "p_DtoM_init must be shape (I,J)"
            self.p_DtoM = P0.copy()

    # ---- components of the mapping Q(p) ----
    def _Q_j(self) -> np.ndarray:
        """Compute effective training revenue per model:
        Q_j = sum_k ω_{jk} * p_{M_j→B_k}."""
        Q = np.zeros(self.J, dtype=float)
        for j, blk in enumerate(self.buyers):
            Q[j] = float(np.dot(blk.omega, self.p_MtoB[j]))
        return Q

    def forward_quote(self) -> List[np.ndarray]:
        """Supply-side quote to buyers:
        v_{M_j→B_k} = κ_{M_j→B_k} + (1 + δ_j) * sum_i p_{D_i→M_j}.
        Returns a list with one (K_j,) array per model j.
        """
        spend_j = np.sum(self.p_DtoM, axis=0)  # (J,)
        v_list: List[np.ndarray] = []
        for j, blk in enumerate(self.buyers):
            vj = blk.kappa_mb + (1.0 + self.delta[j]) * spend_j[j]  # (K_j,)
            v_list.append(vj)
        return v_list

    def backward_quote(self) -> np.ndarray:
        """Demand-side quote to datasets:
        v_{D_i→M_j} = κ_{D_i} + (SV_{i|j}/(1+δ_j)) * Q_j.
        Returns an (I, J) matrix.
        """
        Q = self._Q_j()  # (J,)
        factor = Q / (1.0 + self.delta)  # (J,)
        return self.kappa_D[:, None] + self.SV * factor[None, :]

    def project_boxes(self) -> None:
        """No-op in the no-clipping base; kept for API compatibility."""
        return

    def residual_L2(self, v_M_list: List[np.ndarray], v_D_mat: np.ndarray) -> float:
        """Compute L2 residual: || Q(p) - p ||_2 for monitoring."""
        r2 = 0.0
        for j in range(self.J):
            r2 += float(np.sum((v_M_list[j] - self.p_MtoB[j]) ** 2))
        r2 += float(np.sum((v_D_mat - self.p_DtoM) ** 2))
        return np.sqrt(r2)

    def gap_normalized(self) -> float:
        """Normalized fixed-point gap: ||Q(p) - p||_2 / sqrt(d),
        where d is the total number of scalar prices."""
        vM = self.forward_quote()
        vD = self.backward_quote()
        num = self.residual_L2(vM, vD)
        d = sum(self.K_list) + self.I * self.J
        return num / np.sqrt(d)

    def export(self) -> Dict[str, Any]:
        """Export current prices and normalized gap."""
        return {
            "p_DtoM": self.p_DtoM.copy(),
            "p_MtoB": [p.copy() for p in self.p_MtoB],
            "gap": self.gap_normalized(),
        }
