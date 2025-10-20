from typing import Any, Dict
from pricing.base import _PricingBase


class SupplyFirstPricing(_PricingBase):
    """
    Forward computation (FC) method for one-way price propagation.

    Steps:
        1. Compute model-to-buyer quotations directly from data-side prices and profit margins:
               p_{M→B} := v_{M→B} = κ_{M→B} + (1 + δ) * sum_i p_{D→M}(i, j)
        2. Keep the individual data-side posted prices p_{D→M}.
        3. No bidirectional fixed-point iteration or clipping is applied.
    """

    def fit(self) -> Dict[str, Any]:
        # Compute model-side quotations directly (no clipping or projection)
        vM = self.forward_quote()

        # Assign computed prices to each model's buyer side
        for j in range(self.J):
            self.p_MtoB[j] = vM[j]

        # No projection or clipping applied
        return self.export()
