"""
constraints.py
==============
Builds and validates institutional portfolio constraint sets.

Supported constraints
---------------------
- weights sum to 1
- long-only or long/short
- per-asset minimum / maximum weight
- per-sector maximum exposure
- target portfolio volatility
- maximum turnover from current weights

The module exposes a ConstraintBuilder that produces a standardised
constraint dict consumed by PortfolioOptimizer.optimize().

It also provides standalone validator functions so constraints can be
checked independently of the optimizer.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint builder
# ---------------------------------------------------------------------------

class ConstraintBuilder:
    """
    Fluent builder for portfolio constraint sets.

    Example
    -------
    >>> cb = (ConstraintBuilder(n_assets=5)
    ...       .long_only()
    ...       .max_weight(0.30)
    ...       .min_weight(0.02)
    ...       .target_volatility(0.12)
    ...       .turnover(0.20, current_weights))
    >>> constraints = cb.build()
    """

    def __init__(self, n_assets: int, asset_names: Optional[List[str]] = None):
        """
        Parameters
        ----------
        n_assets : int
            Number of assets in the universe.
        asset_names : list of str, optional
            Asset labels (used for sector mapping and logging).
        """
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1.")
        self.n = n_assets
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(n_assets)]

        # Internal state
        self._long_only: bool = True
        self._min_weight: float = 0.0
        self._max_weight: float = 1.0
        self._per_asset_bounds: Optional[List[Tuple[float, float]]] = None
        self._target_vol: Optional[float] = None
        self._turnover_limit: Optional[float] = None
        self._current_weights: Optional[np.ndarray] = None
        self._sector_constraints: List[Dict] = []

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def long_only(self, value: bool = True) -> "ConstraintBuilder":
        """Enforce non-negative weights (long-only)."""
        self._long_only = value
        return self

    def allow_short(self) -> "ConstraintBuilder":
        """Allow negative weights (long/short)."""
        self._long_only = False
        return self

    def max_weight(self, limit: float) -> "ConstraintBuilder":
        """
        Maximum weight any single asset may hold.

        Parameters
        ----------
        limit : float
            E.g. 0.30 for a 30 % cap.
        """
        if not 0 < limit <= 1.0:
            raise ValueError(f"max_weight must be in (0, 1]; got {limit}.")
        self._max_weight = limit
        return self

    def min_weight(self, limit: float) -> "ConstraintBuilder":
        """
        Minimum weight any single asset must hold (if included).

        Parameters
        ----------
        limit : float
            E.g. 0.01 for a 1 % floor.
        """
        if limit < 0 and self._long_only:
            raise ValueError("min_weight cannot be negative in long-only mode.")
        self._min_weight = limit
        return self

    def per_asset_bounds(
        self, bounds: List[Tuple[float, float]]
    ) -> "ConstraintBuilder":
        """
        Specify (min, max) weight bounds per asset individually.

        Overrides global min_weight / max_weight for the affected assets.

        Parameters
        ----------
        bounds : list of (float, float)
            Length must match n_assets.
        """
        if len(bounds) != self.n:
            raise ValueError(
                f"bounds length ({len(bounds)}) must match n_assets ({self.n})."
            )
        self._per_asset_bounds = bounds
        return self

    def target_volatility(self, vol: float) -> "ConstraintBuilder":
        """
        Maximum annualised portfolio volatility target.

        Parameters
        ----------
        vol : float
            E.g. 0.10 for a 10 % p.a. volatility cap.
        """
        if vol <= 0:
            raise ValueError("target_volatility must be positive.")
        self._target_vol = vol
        return self

    def turnover(
        self, limit: float, current_weights: Union[np.ndarray, List[float]]
    ) -> "ConstraintBuilder":
        """
        Maximum one-way portfolio turnover.

        Parameters
        ----------
        limit : float
            Maximum sum of absolute weight changes (e.g. 0.20 = 20 %).
        current_weights : array-like
            Current portfolio weights (sum to 1).
        """
        if limit < 0:
            raise ValueError("turnover limit must be non-negative.")
        cw = np.asarray(current_weights, dtype=float)
        if len(cw) != self.n:
            raise ValueError("current_weights length must match n_assets.")
        self._turnover_limit = limit
        self._current_weights = cw
        return self

    def add_sector_constraint(
        self,
        sector_assets: List[str],
        max_exposure: float,
        min_exposure: float = 0.0,
    ) -> "ConstraintBuilder":
        """
        Limit total allocation to a group of assets (sector / region / factor).

        Parameters
        ----------
        sector_assets : list of str
            Asset names belonging to the sector.
        max_exposure : float
            Maximum total weight for the sector.
        min_exposure : float
            Minimum total weight for the sector (default 0).
        """
        unknown = [a for a in sector_assets if a not in self.asset_names]
        if unknown:
            logger.warning("Unknown assets in sector constraint: %s", unknown)

        indices = [i for i, a in enumerate(self.asset_names) if a in sector_assets]
        if not indices:
            logger.warning("No valid assets found for sector constraint; skipping.")
            return self

        self._sector_constraints.append(
            {
                "indices": indices,
                "max_exposure": max_exposure,
                "min_exposure": min_exposure,
                "assets": sector_assets,
            }
        )
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Dict:
        """
        Compile all constraints into a dictionary ready for
        ``PortfolioOptimizer.optimize()``.

        Returns
        -------
        dict
            Constraint specification.
        """
        lo = self._min_weight if self._long_only else max(self._min_weight, -1.0)
        hi = self._max_weight

        if self._per_asset_bounds:
            bounds = self._per_asset_bounds
        else:
            bounds = [(lo, hi)] * self.n

        # Scipy-style constraint list
        scipy_cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if self._target_vol is not None:
            # Stored separately; optimizer reads it directly from the dict
            pass

        if self._turnover_limit is not None:
            tl = self._turnover_limit
            cw = self._current_weights
            scipy_cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, t=tl, c=cw: t - np.sum(np.abs(w - c)),
                }
            )

        for sc in self._sector_constraints:
            idx = sc["indices"]
            max_exp = sc["max_exposure"]
            min_exp = sc["min_exposure"]
            scipy_cons.append(
                {"type": "ineq", "fun": lambda w, i=idx, m=max_exp: m - np.sum(w[i])}
            )
            if min_exp > 0:
                scipy_cons.append(
                    {"type": "ineq", "fun": lambda w, i=idx, m=min_exp: np.sum(w[i]) - m}
                )

        constraint_dict: Dict = {
            "long_only": self._long_only,
            "min_weight": lo,
            "max_weight": hi,
            "bounds": bounds,
            "scipy_constraints": scipy_cons,
        }

        if self._target_vol is not None:
            constraint_dict["target_volatility"] = self._target_vol

        if self._turnover_limit is not None:
            constraint_dict["turnover_limit"] = self._turnover_limit
            constraint_dict["current_weights"] = self._current_weights

        if self._sector_constraints:
            constraint_dict["sector_constraints"] = self._sector_constraints

        logger.debug("Constraints built: %s", list(constraint_dict.keys()))
        return constraint_dict


# ---------------------------------------------------------------------------
# Standalone validators
# ---------------------------------------------------------------------------

def validate_weights(
    weights: np.ndarray,
    constraints: Dict,
    tol: float = 1e-4,
) -> Tuple[bool, List[str]]:
    """
    Check whether a weight vector satisfies a constraint dict.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights to validate.
    constraints : dict
        Constraint dict produced by ConstraintBuilder.build().
    tol : float
        Numerical tolerance (default 1e-4).

    Returns
    -------
    (bool, list of str)
        True if all constraints are satisfied, plus a list of violation messages.
    """
    violations = []
    w = np.asarray(weights, dtype=float)

    # Sum-to-one
    if abs(w.sum() - 1.0) > tol:
        violations.append(f"Weights sum to {w.sum():.6f}, expected 1.0")

    # Long-only
    if constraints.get("long_only", True) and np.any(w < -tol):
        violations.append(f"Negative weights found in long-only mode: {w[w < 0]}")

    # Per-asset bounds
    bounds = constraints.get("bounds", [])
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None and w[i] < lo - tol:
            violations.append(f"Asset {i} weight {w[i]:.4f} < min {lo}")
        if hi is not None and w[i] > hi + tol:
            violations.append(f"Asset {i} weight {w[i]:.4f} > max {hi}")

    # Sector constraints
    for sc in constraints.get("sector_constraints", []):
        total = w[sc["indices"]].sum()
        if total > sc["max_exposure"] + tol:
            violations.append(
                f"Sector {sc['assets']} exposure {total:.4f} > max {sc['max_exposure']}"
            )
        if total < sc["min_exposure"] - tol:
            violations.append(
                f"Sector {sc['assets']} exposure {total:.4f} < min {sc['min_exposure']}"
            )

    # Turnover
    if "turnover_limit" in constraints and "current_weights" in constraints:
        turnover = np.sum(np.abs(w - constraints["current_weights"]))
        if turnover > constraints["turnover_limit"] + tol:
            violations.append(
                f"Turnover {turnover:.4f} exceeds limit {constraints['turnover_limit']}"
            )

    is_valid = len(violations) == 0
    if not is_valid:
        for v in violations:
            logger.warning("Constraint violation: %s", v)
    return is_valid, violations


def describe_constraints(constraints: Dict, asset_names: Optional[List[str]] = None) -> str:
    """
    Return a human-readable summary of the active constraints.

    Parameters
    ----------
    constraints : dict
        Constraint dict from ConstraintBuilder.build().
    asset_names : list of str, optional
        Asset labels.

    Returns
    -------
    str
        Multi-line description.
    """
    lines = ["=== Portfolio Constraints ==="]
    lines.append(f"  Long-only           : {constraints.get('long_only', True)}")
    lines.append(f"  Min weight          : {constraints.get('min_weight', 0.0):.2%}")
    lines.append(f"  Max weight          : {constraints.get('max_weight', 1.0):.2%}")

    if "target_volatility" in constraints:
        lines.append(f"  Target volatility   : {constraints['target_volatility']:.2%}")

    if "turnover_limit" in constraints:
        lines.append(f"  Turnover limit      : {constraints['turnover_limit']:.2%}")

    for sc in constraints.get("sector_constraints", []):
        names = sc.get("assets", sc["indices"])
        lines.append(
            f"  Sector {names}: [{sc['min_exposure']:.2%}, {sc['max_exposure']:.2%}]"
        )

    return "\n".join(lines)
