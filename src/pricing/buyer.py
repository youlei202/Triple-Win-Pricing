from dataclasses import dataclass
import numpy as np
from typing import List, Optional


@dataclass
class BuyerBlock:
    """
    Per-model buyer block.
    Arrays are length-K_j for model j.
    - kappa_mb:  κ_{M_j→B_k} > 0
    - p0:        lower bound p_{M_j→B_k}^0
    - R:         upper bound R_{M_j}^{(k)}
    - omega:     weights ω_{jk} in Q_j = sum_k ω_{jk} p_{M_j→B_k}
                 (must be nonnegative; sum ≤ 1 is typical)
    - p_init:    optional initial prices
    """

    kappa_mb: np.ndarray
    p0: np.ndarray
    R: np.ndarray
    omega: np.ndarray
    p_init: Optional[np.ndarray] = None
