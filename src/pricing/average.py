from typing import Any, Dict, List, Optional
from pricing.base import _PricingBase
import numpy as np


class AVGPricing(_PricingBase):
    """
    Classic averaged fixed-point iteration:
        p^{t+1} = (1 - γ) * p^{t} + γ * I(p^{t})

    This version performs no clipping or projection to any bounds.
    Iteration stops when both the residual and maximum change fall below tol,
    or when the maximum number of iterations is reached.
    """

    def __init__(self, *args, gamma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0.0 < gamma <= 1.0, "gamma must be in (0, 1]"
        self.gamma = float(gamma)

    def fit(self) -> Dict[str, Any]:
        prev_res = np.inf

        for it in range(1, self.max_iter + 1):
            # Forward and backward quotations
            vM = self.forward_quote()
            vD = self.backward_quote()

            # Compute residual ||I(p^t) - p^t||
            res = self.residual_L2(vM, vD)

            # Averaged update (no projection)
            for j in range(self.J):
                self.p_MtoB[j] = (1.0 - self.gamma) * self.p_MtoB[j] + self.gamma * vM[
                    j
                ]
            self.p_DtoM = (1.0 - self.gamma) * self.p_DtoM + self.gamma * vD

            # Convergence check: residual and maximum change
            maxchg = 0.0
            for j in range(self.J):
                maxchg = max(maxchg, float(np.max(np.abs(self.p_MtoB[j] - vM[j]))))
            maxchg = max(maxchg, float(np.max(np.abs(self.p_DtoM - vD))))

            if self.verbose and (it % 50 == 0 or it == 1):
                print(f"[AVG iter {it:5d}] residual={res:.3e}  max-change={maxchg:.3e}")

            if res < self.tol and maxchg < self.tol:
                break
            prev_res = res

        # Return final state
        return self.export()


class TrackingAVGPricing(AVGPricing):
    """
    Averaged fixed-point iteration with tracking.
      - res_hist[t]    = || I(p^t) - p^t ||_2   (residual before the update)
      - maxchg_hist[t] = max change between p^{t+1} and p^t (after the update)
    No clipping / projection is performed (inherits no-clip AVG).
    """

    def fit(self) -> Dict[str, Any]:
        self.res_hist = []
        self.maxchg_hist = []

        prev_res = np.inf

        for it in range(1, self.max_iter + 1):
            # Compute quotations from current prices p^t
            vM = self.forward_quote()
            vD = self.backward_quote()

            # Residual BEFORE update
            res = self.residual_L2(vM, vD)
            self.res_hist.append(float(res))

            # Keep old prices
            old_M = [p.copy() for p in self.p_MtoB]
            old_D = self.p_DtoM.copy()

            # Averaged update (no projection)
            for j in range(self.J):
                self.p_MtoB[j] = (1.0 - self.gamma) * self.p_MtoB[j] + self.gamma * vM[
                    j
                ]
            self.p_DtoM = (1.0 - self.gamma) * self.p_DtoM + self.gamma * vD

            # Max change AFTER update
            maxchg = 0.0
            for j in range(self.J):
                maxchg = max(maxchg, float(np.max(np.abs(self.p_MtoB[j] - old_M[j]))))
            maxchg = max(maxchg, float(np.max(np.abs(self.p_DtoM - old_D))))
            self.maxchg_hist.append(float(maxchg))

            if self.verbose and (it % 50 == 0 or it == 1):
                print(f"[AVG iter {it:5d}] residual={res:.3e}  max-change={maxchg:.3e}")

            # Stopping rule
            if res < self.tol and maxchg < self.tol:
                break
            prev_res = res

        out = self.export()
        out["res_hist"] = np.array(self.res_hist, dtype=float)
        out["maxchg_hist"] = np.array(self.maxchg_hist, dtype=float)
        out["iterations"] = it
        out["residual"] = float(res)
        return out
