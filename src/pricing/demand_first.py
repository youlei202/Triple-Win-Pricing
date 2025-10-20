from typing import Any, Dict, List, Optional
from pricing.base import _PricingBase


class DemandFirstPricing(_PricingBase):
    """
    Direct computation (DC) method for one-way price propagation.

    Step:
        - Compute data-side compensation based on buyer-side prices and producer margins:
              p_{D→M} := v_{D→M} = κ_D + (SV / (1 + δ)) * Q_j
        - Keep the model-side individual posted prices p_{M→B} (initialized from p0).
        - No bidirectional fixed-point iteration is performed.

    This version does not perform any clipping or projection to bounds.
    """

    def fit(self) -> Dict[str, Any]:
        # Directly compute backward quote without any clipping
        vD = self.backward_quote()
        self.p_DtoM = vD
        # No projection applied
        return self.export()
