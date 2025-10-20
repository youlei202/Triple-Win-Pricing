from typing import Any, Dict, List, Optional
from pricing.base import _PricingBase
import numpy as np


class BrokerCentricPricing(_PricingBase):
    """
    Broker-centric (or producer-led) heuristic pricing.

    The platform or producer acts as a leader and selects a markup rate α_j
    so that the model prices approach the buyers' upper limits (or an aggregate target).

    Heuristic procedure:
        1. Compute total data spending for each model:
               spend_j = sum_i p_{D→M}(i, j)
        2. Define a target price scale for model j:
               R_hat = min_k R_{M_j}^{(k)}   (can also use a weighted average)
        3. Estimate the markup rate:
               α_j = max{0, (R_hat - mean_k κ_{M_j→B_k}) / spend_j - 1}
        4. Compute model-to-buyer prices:
               p_{M_j→B_k} = κ_{M_j→B_k} + (1 + α_j) * spend_j
        (No clipping is applied.)
    """

    def fit(self) -> Dict[str, Any]:
        # Compute total data-side spending per model
        spend_j = np.sum(self.p_DtoM, axis=0)  # shape (J,)

        for j, blk in enumerate(self.buyers):
            # Choose a target upper limit (can also use np.average with weights=omega)
            R_hat = float(np.min(blk.R))
            kappa_mean = float(np.mean(blk.kappa_mb))

            if spend_j[j] <= 1e-12:
                alpha = 0.0
            else:
                alpha = max(0.0, (R_hat - kappa_mean) / spend_j[j] - 1.0)

            # Apply α_j without clipping
            self.p_MtoB[j] = blk.kappa_mb + (1.0 + alpha) * spend_j[j]

        # No projection or clipping applied
        return self.export()



class BrokerCentricPricingRobust(_PricingBase):
    """
    Robust Broker-Centric (single pass, no clipping, no iteration).
    """
    def __init__(
        self,
        *args,
        q_upper: float = 0.6,
        lam: float = 0.4,
        beta: float = 0.5,
        q_aff: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # === Tunables (with expected qualitative effects) ===

        # q_upper : float in [0,1]
        #   - Robust ceiling quantile used for R_hat.
        #   - ↑ q_upper → uses higher buyer quantile as ceiling → higher α_j, higher prices.
        #   - ↓ q_upper → ignores rich outliers → lower α_j, safer / more conservative pricing.
        self.q_upper = float(q_upper)

        # lam : float in [0,1]
        #   - Shrink weight toward producer margin δ_j.
        #   - lam = 0 → trust δ_j only (supply-driven).
        #   - lam = 1 → trust heuristic α_raw (demand-driven).
        #   - ↑ lam → more responsive to buyer limits, but risk of overpricing.
        self.lam = float(lam)

        # beta : non-negative float
        #   - Cap on α_j above δ_j (i.e., α_j ≤ δ_j + β).
        #   - ↑ beta → allow higher markups → riskier but higher potential profit.
        #   - ↓ beta → more conservative, smaller markup range.
        self.beta = float(beta)

        # q_aff : float in [0,1]
        #   - Affordability quantile for one-shot scaling.
        #   - Ensures at least q_aff fraction of buyers can afford the model.
        #   - ↓ q_aff (e.g., 0.3) → stronger downward correction (stricter affordability).
        #   - ↑ q_aff (e.g., 0.7) → weaker correction (looser affordability).
        self.q_aff = float(q_aff)

    def fit(self) -> Dict[str, Any]:
        spend_j = np.sum(self.p_DtoM, axis=0)  # (J,)

        for j, blk in enumerate(self.buyers):
            # 1) Robust ceiling for demand
            R_hat = float(np.quantile(blk.R, self.q_upper))
            kappa_mean = float(np.mean(blk.kappa_mb))

            # 2) Raw alpha from ceiling (guard tiny spend)
            if spend_j[j] <= 1e-12:
                alpha_raw = 0.0
            else:
                alpha_raw = max(0.0, (R_hat - kappa_mean) / spend_j[j] - 1.0)

            # 3) Shrink towards producer margin δ_j and hard-cap
            alpha = (1.0 - self.lam) * self.delta[j] + self.lam * alpha_raw
            alpha = min(alpha, self.delta[j] + self.beta)

            # 4) Forward price (no clipping)
            self.p_MtoB[j] = blk.kappa_mb + (1.0 + alpha) * spend_j[j]

            # 5) One-shot affordability scaling (keep kappa part intact)
            avg_p = float(np.mean(self.p_MtoB[j]))
            R_q = float(np.quantile(blk.R, self.q_aff))
            if avg_p > 1e-12 and R_q < avg_p:
                s = max(0.0, min(1.0, R_q / avg_p))
                self.p_MtoB[j] = blk.kappa_mb + s * (self.p_MtoB[j] - blk.kappa_mb)

        return self.export()