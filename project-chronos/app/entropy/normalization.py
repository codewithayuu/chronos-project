"""
SampEn normalization parameters.

These min/max ranges define the expected SampEn range for each vital sign
in ICU populations. Derived from literature values for m=2, r=0.2*SD, N≈300.

Used to normalize raw SampEn values to [0, 1] for the Composite Entropy Score.
  - 0.0 = minimum expected complexity (very regular / concerning)
  - 1.0 = maximum expected complexity (healthy variability)
"""

import numpy as np
from typing import Optional


# ──────────────────────────────────────────────
# Reference ranges (calibrated from literature)
# ──────────────────────────────────────────────
# These can be updated with actual MIMIC-IV population statistics.
# For now, based on published SampEn values in ICU studies:
#   - Costa et al. (2005), Ahmad et al. (2009), Lake et al. (2002)

SAMPEN_RANGES = {
    "heart_rate": {"min": 0.10, "max": 2.50},
    "spo2": {"min": 0.05, "max": 2.00},
    "bp_systolic": {"min": 0.10, "max": 2.20},
    "bp_diastolic": {"min": 0.10, "max": 2.00},
    "resp_rate": {"min": 0.05, "max": 2.30},
    "temperature": {"min": 0.05, "max": 1.80},
}


def normalize_sampen(value: float, vital_name: str) -> Optional[float]:
    """
    Normalize a raw SampEn value to [0, 1] using population reference ranges.

    Parameters
    ----------
    value : float
        Raw SampEn value.
    vital_name : str
        Name of the vital sign (must match keys in SAMPEN_RANGES).

    Returns
    -------
    float or None
        Normalized value in [0, 1], or None if input is NaN.
    """
    if value is None or np.isnan(value):
        return None

    ranges = SAMPEN_RANGES.get(vital_name)
    if ranges is None:
        return 0.5  # unknown vital, assume mid-range

    span = ranges["max"] - ranges["min"]
    if span <= 0:
        return 0.5

    normalized = (value - ranges["min"]) / span

    # Clamp to [0, 1]
    return float(max(0.0, min(1.0, normalized)))


def get_ranges():
    """Return a copy of the normalization ranges (for inspection/debugging)."""
    return {k: dict(v) for k, v in SAMPEN_RANGES.items()}
