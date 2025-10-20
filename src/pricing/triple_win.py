# pricing/triple_win.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from pricing.base import _PricingBase


class TripleWinPricing(_PricingBase):
    """
    Raw fixed-point iteration (no clipping):
        p^{t+1} = Q(p^t)
    Stops when both the residual and the max change fall below tol.
    """

    def fit(self) -> Dict[str, Any]:
        prev_res = np.inf

        for it in range(1, self.max_iter + 1):
            # Quotes from current prices p^t
            v_M_list = self.forward_quote()  # list of (K_j,)
            v_D_mat = self.backward_quote()  # (I, J)

            # Residual BEFORE update: ||Q(p^t) - p^t||_2
            res = self.residual_L2(v_M_list, v_D_mat)

            # Update p^{t+1} ← Q(p^t)  (no projection)
            for j in range(self.J):
                self.p_MtoB[j] = v_M_list[j]
            self.p_DtoM = v_D_mat

            # Max change AFTER update: max |p^{t+1} - Q(p^t)| = 0 here,
            # but we still compute versus the quotes for symmetry/clarity.
            maxchg = 0.0
            for j in range(self.J):
                maxchg = max(
                    maxchg, float(np.max(np.abs(self.p_MtoB[j] - v_M_list[j])))
                )
            maxchg = max(maxchg, float(np.max(np.abs(self.p_DtoM - v_D_mat))))

            if self.verbose and (it % 50 == 0 or it == 1):
                print(f"[iter {it:5d}] residual={res:.3e}  max-change={maxchg:.3e}")

            if res < self.tol and maxchg < self.tol:
                break

            prev_res = res

        out = self.export()
        out.update({"iterations": it, "residual": float(res)})
        return out


class TrackingTripleWinPricing(TripleWinPricing):
    """
    Same as TripleWinPricing (no clipping), but records per-iteration history:
      - res_hist[t]    = ||Q(p^t) - p^t||_2   (residual before update)
      - maxchg_hist[t] = max |p^{t+1} - p^{t}| (after update)
    """

    def fit(self) -> Dict[str, Any]:
        self.res_hist: List[float] = []
        self.maxchg_hist: List[float] = []

        prev_res = np.inf

        for it in range(1, self.max_iter + 1):
            v_M_list = self.forward_quote()
            v_D_mat = self.backward_quote()

            # Residual BEFORE update
            res = self.residual_L2(v_M_list, v_D_mat)
            self.res_hist.append(float(res))

            # Save old prices p^t
            old_M = [p.copy() for p in self.p_MtoB]
            old_D = self.p_DtoM.copy()

            # Update p^{t+1} ← Q(p^t)  (no projection)
            for j in range(self.J):
                self.p_MtoB[j] = v_M_list[j]
            self.p_DtoM = v_D_mat

            # Max change AFTER update: max |p^{t+1} - p^{t}|
            maxchg = 0.0
            for j in range(self.J):
                maxchg = max(maxchg, float(np.max(np.abs(self.p_MtoB[j] - old_M[j]))))
            maxchg = max(maxchg, float(np.max(np.abs(self.p_DtoM - old_D))))
            self.maxchg_hist.append(float(maxchg))

            if self.verbose and (it % 50 == 0 or it == 1):
                print(f"[iter {it:5d}] residual={res:.3e}  max-change={maxchg:.3e}")

            if res < self.tol and maxchg < self.tol:
                break

            prev_res = res

        out = self.export()
        out.update(
            {
                "iterations": it,
                "residual": float(res),
                "res_hist": np.array(self.res_hist, dtype=float),
                "maxchg_hist": np.array(self.maxchg_hist, dtype=float),
            }
        )
        return out
