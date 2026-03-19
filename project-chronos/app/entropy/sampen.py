"""
Sample Entropy (SampEn) computation.

SampEn measures the complexity/irregularity of a time series.
  - Higher SampEn → more complex, unpredictable → HEALTHIER
  - Lower SampEn  → more regular, predictable  → CONCERNING

Two implementations:
  1. Numba JIT-compiled (if numba is installed) — ~5ms for N=300
  2. NumPy vectorized fallback — ~50-100ms for N=300
"""

import numpy as np

# ──────────────────────────────────────────────
# Try importing Numba for JIT compilation
# ──────────────────────────────────────────────

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


# ──────────────────────────────────────────────
# Implementation 1: Numba JIT (preferred)
# ──────────────────────────────────────────────

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _count_matches_numba(data, m, r, N):
        """
        Count template matches for embedding dimensions m and m+1.
        Numba JIT-compiled for maximum performance.
        """
        B = 0  # matches of length m
        A = 0  # matches of length m+1

        for i in range(N - m):
            for j in range(i + 1, N - m):
                # Check m-length match (Chebyshev distance)
                match = True
                for k in range(m):
                    if abs(data[i + k] - data[j + k]) > r:
                        match = False
                        break

                if match:
                    B += 1
                    # Check (m+1)-length match
                    if abs(data[i + m] - data[j + m]) <= r:
                        A += 1

        return A, B


# ──────────────────────────────────────────────
# Implementation 2: NumPy vectorized (fallback)
# ──────────────────────────────────────────────

def _count_matches_numpy(data, m, r, N):
    """
    Count template matches using NumPy vectorization.
    No Numba required. Slower but works everywhere.
    """
    n = N - m  # number of templates
    if n <= 1:
        return 0, 0

    # Build all templates of length m+1 using stride tricks
    # templates[i] = data[i : i+m+1]
    strides = data.strides[0]
    shape = (n, m + 1)
    strides_2d = (strides, strides)
    templates = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides_2d)

    templates_m = templates[:, :m]  # m-length templates

    B_total = 0
    A_total = 0

    for i in range(n - 1):
        # Chebyshev distance between template[i] and all templates[j] where j > i
        diffs = np.abs(templates_m[i + 1:] - templates_m[i])
        max_diffs = np.max(diffs, axis=1)
        matches = max_diffs <= r

        b_count = int(np.sum(matches))
        B_total += b_count

        if b_count > 0:
            # For matching m-length templates, check the (m+1)-th element
            extra_diffs = np.abs(templates[i + 1:, m][matches] - templates[i, m])
            A_total += int(np.sum(extra_diffs <= r))

    return A_total, B_total


# ──────────────────────────────────────────────
# Select the best available implementation
# ──────────────────────────────────────────────

if _NUMBA_AVAILABLE:
    _count_matches = _count_matches_numba
    _BACKEND = "numba"
else:
    _count_matches = _count_matches_numpy
    _BACKEND = "numpy"


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r_fraction: float = 0.2
) -> float:
    """
    Compute Sample Entropy (SampEn) of a time series.

    Parameters
    ----------
    data : np.ndarray
        1D array of time series values. NaN values are removed automatically.
    m : int
        Embedding dimension (template length). Default: 2.
    r_fraction : float
        Tolerance as a fraction of the time series standard deviation. Default: 0.2.

    Returns
    -------
    float
        SampEn value. Higher = more complex. Returns NaN if computation is not possible.
        Returns 0.0 if the signal is constant (zero complexity).
    """
    # Clean input
    data = np.asarray(data, dtype=np.float64).ravel()
    data = data[~np.isnan(data)]

    N = len(data)
    if N < (m + 2):
        return float("nan")  # insufficient data

    # Standard deviation
    sd = np.std(data)
    if sd < 1e-10:
        return 0.0  # constant signal → zero complexity

    # Tolerance
    r = r_fraction * sd

    # Count matches
    A, B = _count_matches(data, m, r, N)

    # Compute SampEn
    if B == 0:
        return float("nan")  # no m-length matches (very noisy or very short)
    if A == 0:
        return float("nan")  # undefined (would be infinity)

    return float(-np.log(A / B))


def get_backend() -> str:
    """Return which computation backend is active."""
    return _BACKEND
