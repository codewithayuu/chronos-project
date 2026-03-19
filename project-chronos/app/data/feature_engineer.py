"""
Feature Engineer — computes the 45-feature vector from patient state.

Used by Person 3's pipeline integration to feed ML models.
Input: rolling vitals window, entropy state, drug state, demographics.
Output: np.ndarray of shape (45,) — never raises, returns zeros + NaN flags on error.

Feature layout (indices 0-44):
 0  age
 1  hr_current, 2 hr_mean_6h, 3 hr_min_6h, 4 hr_std_6h, 5 hr_trend_6h
 6  bp_sys_current, 7 bp_sys_mean_6h, 8 bp_sys_min_6h, 9 bp_sys_std_6h, 10 bp_sys_trend_6h
11  rr_current, 12 rr_mean_6h, 13 rr_std_6h, 14 spo2_current, 15 spo2_mean_6h,
16  spo2_min_6h, 17 spo2_std_6h
18  sampen_hr, 19 sampen_bp_sys, 20 sampen_rr, 21 sampen_spo2
22  ces_adjusted, 23 ces_raw, 24 ces_slope_6h, 25 ces_velocity
26  entropy_vital_divergence, 27 mse_slope_index
28  shock_index, 29 map_current, 30 pulse_pressure, 31 rr_spo2_ratio
32  vasopressor_active, 33 sedative_active, 34 beta_blocker_active, 35 opioid_active
36  total_vasoactive_dose, 37 inotrope_active, 38 num_active_drugs
39  temp_current, 40 temp_deviation
41  window_fill_fraction, 42 time_since_drug_change
43  hr_bp_correlation, 44 entropy_drug_interaction
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional


# Norepinephrine-equivalent conversion factors
_NE_EQUIV = {
    "norepinephrine": 1.0,
    "vasopressin": 0.4,       # 2.5 units vasopressin = 1 NE eq
    "dopamine": 0.067,        # 15 mcg/kg/min = 1 NE eq
    "phenylephrine": 0.1,     # 10 mcg/min = 1 NE eq
    "epinephrine": 1.0,
}


class FeatureEngineer:
    """
    Computes the 45-feature vector from patient state.

    Contract (Person 3 interface):
        - __init__(population_stats_path: str)
        - compute_features(vitals_window, entropy_state, drug_state, demographics) -> np.ndarray(45,)
        - impute_warmup(features: np.ndarray) -> np.ndarray
    """

    NUM_FEATURES = 45

    def __init__(self, population_stats_path: str = "data/ml/population_stats.json"):
        self._pop_stats = None
        try:
            p = Path(population_stats_path)
            if p.exists():
                with open(p) as f:
                    self._pop_stats = json.load(f)
        except Exception:
            pass

        # Defaults if population stats not available
        if self._pop_stats is None:
            self._pop_stats = {
                "sampen_hr_median": 1.2,
                "sampen_bp_sys_median": 0.9,
                "sampen_rr_median": 1.0,
                "sampen_spo2_median": 0.7,
                "ces_median": 0.65,
            }

    def compute_features(
        self,
        vitals_window: dict,
        entropy_state: dict,
        drug_state: dict,
        demographics: dict,
    ) -> np.ndarray:
        """
        Compute 45-feature vector. Never raises — returns zeros on failure.

        Parameters
        ----------
        vitals_window : dict
            Rolling window of vital-sign arrays, e.g.
            {'hr': [80, 82, ...], 'bp_sys': [...], 'bp_dia': [...],
             'rr': [...], 'spo2': [...], 'temp': [...]}
        entropy_state : dict
            From entropy engine: ces_adjusted, ces_raw, ces_slope_6h,
            sampen_hr, sampen_bp_sys, sampen_rr, sampen_spo2, window_size, etc.
        drug_state : dict
            drug_masking (bool), active_drugs (list of dicts),
            each with drug_name, drug_class, dose, unit.
        demographics : dict
            age, sex, weight_kg.

        Returns
        -------
        np.ndarray of shape (45,)
        """
        try:
            return self._compute_features_impl(
                vitals_window, entropy_state, drug_state, demographics
            )
        except Exception:
            return np.zeros(self.NUM_FEATURES)

    def _compute_features_impl(
        self, vitals_window, entropy_state, drug_state, demographics
    ) -> np.ndarray:
        features = np.zeros(self.NUM_FEATURES)

        # ── Demographics (0) ──
        features[0] = float(demographics.get("age", 60))

        # ── Heart rate features (1-5) ──
        hr = np.array(vitals_window.get("hr", []), dtype=float)
        if len(hr) > 0:
            features[1] = hr[-1]                     # hr_current
            features[2] = np.nanmean(hr)             # hr_mean_6h
            features[3] = np.nanmin(hr)              # hr_min_6h
            features[4] = np.nanstd(hr)              # hr_std_6h
            features[5] = self._linear_slope(hr)     # hr_trend_6h

        # ── BP systolic features (6-10) ──
        bp = np.array(vitals_window.get("bp_sys", []), dtype=float)
        if len(bp) > 0:
            features[6] = bp[-1]
            features[7] = np.nanmean(bp)
            features[8] = np.nanmin(bp)
            features[9] = np.nanstd(bp)
            features[10] = self._linear_slope(bp)

        # ── Respiratory rate features (11-13) ──
        rr = np.array(vitals_window.get("rr", []), dtype=float)
        if len(rr) > 0:
            features[11] = rr[-1]
            features[12] = np.nanmean(rr)
            features[13] = np.nanstd(rr)

        # ── SpO2 features (14-17) ──
        spo2 = np.array(vitals_window.get("spo2", []), dtype=float)
        if len(spo2) > 0:
            features[14] = spo2[-1]
            features[15] = np.nanmean(spo2)
            features[16] = np.nanmin(spo2)
            features[17] = np.nanstd(spo2)

        # ── Entropy features (18-27) ──
        features[18] = float(entropy_state.get("sampen_hr", 0) or 0)
        features[19] = float(entropy_state.get("sampen_bp_sys", 0) or 0)
        features[20] = float(entropy_state.get("sampen_rr", 0) or 0)
        features[21] = float(entropy_state.get("sampen_spo2", 0) or 0)
        features[22] = float(entropy_state.get("ces_adjusted", 0.65) or 0.65)
        features[23] = float(entropy_state.get("ces_raw", 0.65) or 0.65)
        features[24] = float(entropy_state.get("ces_slope_6h", 0) or 0)
        # CES velocity = rate of change in CES slope (second derivative)
        features[25] = 0.0  # Requires historical slope data, default 0

        # Entropy vital divergence = std of the 4 SampEn values
        sampen_vals = [features[18], features[19], features[20], features[21]]
        nonzero_sampens = [v for v in sampen_vals if v > 0]
        if len(nonzero_sampens) >= 2:
            features[26] = float(np.std(nonzero_sampens))
        else:
            features[26] = 0.0

        # MSE slope index (slope of SampEn across scales — approximate with single-scale)
        features[27] = 0.0  # Requires multi-scale computation, default 0

        # ── Derived vital features (28-31) ──
        hr_current = features[1] if features[1] > 0 else 80.0
        bp_sys_current = features[6] if features[6] > 0 else 120.0
        bp_dia = np.array(vitals_window.get("bp_dia", []), dtype=float)
        bp_dia_current = bp_dia[-1] if len(bp_dia) > 0 else 80.0

        # Shock index = HR / SBP
        features[28] = hr_current / max(bp_sys_current, 1.0)
        # MAP = (SBP + 2*DBP) / 3
        features[29] = (bp_sys_current + 2 * bp_dia_current) / 3.0
        # Pulse pressure = SBP - DBP
        features[30] = bp_sys_current - bp_dia_current
        # RR/SpO2 ratio
        rr_current = features[11] if features[11] > 0 else 16.0
        spo2_current = features[14] if features[14] > 0 else 97.0
        features[31] = rr_current / max(spo2_current, 1.0)

        # ── Drug features (32-38) ──
        active_drugs = drug_state.get("active_drugs", [])
        drug_classes = set()
        total_ne_dose = 0.0
        for d in active_drugs:
            dclass = (d.get("drug_class") or "").lower()
            dname = (d.get("drug_name") or "").lower()
            dose = float(d.get("dose", 0) or 0)
            drug_classes.add(dclass)

            # Vasopressor NE equivalent
            for vp_name, factor in _NE_EQUIV.items():
                if vp_name in dname:
                    total_ne_dose += dose * factor
                    break

        features[32] = 1.0 if "vasopressor" in drug_classes else 0.0
        features[33] = 1.0 if "sedative" in drug_classes else 0.0
        features[34] = 1.0 if "beta_blocker" in drug_classes else 0.0
        features[35] = 1.0 if "opioid" in drug_classes else 0.0
        features[36] = total_ne_dose
        features[37] = 1.0 if "inotrope" in drug_classes else 0.0
        features[38] = float(len(active_drugs))

        # ── Temperature features (39-40) ──
        temp = np.array(vitals_window.get("temp", []), dtype=float)
        if len(temp) > 0:
            features[39] = temp[-1]
            features[40] = abs(temp[-1] - 37.0)  # deviation from normal

        # ── System features (41-44) ──
        window_size = float(entropy_state.get("window_size", 0) or 0)
        features[41] = min(window_size / 300.0, 1.0)  # window fill fraction
        features[42] = 0.0  # time since last drug change (default 0)

        # HR-BP correlation (from windowed data)
        if len(hr) >= 10 and len(bp) >= 10:
            min_len = min(len(hr), len(bp))
            try:
                corr = np.corrcoef(hr[-min_len:], bp[-min_len:])[0, 1]
                features[43] = corr if np.isfinite(corr) else 0.0
            except Exception:
                features[43] = 0.0

        # Entropy-drug interaction
        if drug_state.get("drug_masking", False):
            features[44] = features[22] * features[38]  # CES × num_drugs
        else:
            features[44] = 0.0

        # Replace any NaN
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def impute_warmup(self, features: np.ndarray) -> np.ndarray:
        """
        Replace entropy features (indices 18-27) with population medians
        during warm-up period when entropy engine hasn't calibrated yet.
        """
        result = features.copy()
        ps = self._pop_stats

        result[18] = ps.get("sampen_hr_median", 1.2)
        result[19] = ps.get("sampen_bp_sys_median", 0.9)
        result[20] = ps.get("sampen_rr_median", 1.0)
        result[21] = ps.get("sampen_spo2_median", 0.7)
        result[22] = ps.get("ces_median", 0.65)
        result[23] = ps.get("ces_median", 0.65)
        result[24] = 0.0  # slope = 0 during warmup
        result[25] = 0.0  # velocity = 0
        result[26] = 0.0  # divergence = 0
        result[27] = 0.0  # MSE slope = 0

        return result

    @staticmethod
    def _linear_slope(arr: np.ndarray) -> float:
        """Compute simple linear regression slope over an array."""
        n = len(arr)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = arr.mean()
        num = np.sum((x - x_mean) * (arr - y_mean))
        den = np.sum((x - x_mean) ** 2)
        if abs(den) < 1e-12:
            return 0.0
        return float(num / den)
