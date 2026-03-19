"""
Multi-Scale Entropy (MSE) computation.

MSE applies Sample Entropy at multiple time scales via coarse-graining.
A healthy signal maintains high entropy across ALL scales.
A deteriorating signal loses complexity at higher scales first.
"""

import numpy as np
from typing import List, Optional
from .sampen import sample_entropy


def coarse_grain(data: np.ndarray, scale: int) -> np.ndarray:
    """
    Coarse-grain a time series at the given scale factor.

    Divides the series into non-overlapping windows of size `scale` 
    and replaces each window with its mean.

    Parameters
    ----------
    data : np.ndarray
        1D time series (NaN-free).
    scale : int
        Coarse-graining factor. scale=1 returns the original series.

    Returns
    -------
    np.ndarray
        Coarse-grained series of length floor(N / scale).
    """
    if scale == 1:
        return data.copy()

    N = len(data)
    n_segments = N // scale
    if n_segments == 0:
        return np.array([], dtype=np.float64)

    trimmed = data[: n_segments * scale]
    reshaped = trimmed.reshape(n_segments, scale)
    return np.mean(reshaped, axis=1)


def multiscale_entropy(
    data: np.ndarray,
    scales: List[int],
    m: int = 2,
    r_fraction: float = 0.2,
    min_points: int = 20,
) -> List[Optional[float]]:
    """
    Compute Multi-Scale Entropy (MSE) of a time series.

    Parameters
    ----------
    data : np.ndarray
        1D time series. NaN values are removed automatically.
    scales : list of int
        Scale factors to compute (e.g., [1, 2, 3, ..., 10]).
    m : int
        Embedding dimension for SampEn.
    r_fraction : float
        Tolerance fraction for SampEn.
    min_points : int
        Minimum points required in coarse-grained series. Default: 20.

    Returns
    -------
    list of float or None
        SampEn at each scale. None if computation failed at that scale.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    data = data[~np.isnan(data)]

    results = []
    for scale in scales:
        cg = coarse_grain(data, scale)

        if len(cg) < max(m + 2, min_points):
            results.append(None)
        else:
            val = sample_entropy(cg, m, r_fraction)
            results.append(val if not np.isnan(val) else None)

    return results
