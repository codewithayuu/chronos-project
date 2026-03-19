"""
Decision Fusion — Core Runtime for Project Chronos.

Combines entropy-based CES scores with ML predictions, drug context,
and NEWS2 clinical scores into a single Final Risk Score (FRS 0-100).

Implements all three bug fixes from the PRD:
  Bug Fix 1: fuse() accepts ml_risk_1h/4h/8h (not just 4h)
  Bug Fix 2: override_reason scoped at method top (not 'in dir()' check)
  Bug Fix 3: (in predictor.py — per-prediction contribs)

Key design decisions:
  - When ML is unavailable (warmup / missing models), weights are
    redistributed from ML to entropy + NEWS2 so FRS never stalls.
  - Override rules ensure entropy CRITICAL is always surfaced.
  - Disagreement detection triggers when entropy and ML diverge significantly.
"""

from typing import Optional, Dict


class DecisionFusion:
    """
    Multi-signal decision fusion engine.

    Merges five risk dimensions into a Final Risk Score (0-100)
    and corresponding severity tier.

    Thresholds (FRS → Severity):
      0-25  → NONE
      26-45 → WATCH
      46-70 → WARNING
      71+   → CRITICAL
    """

    # Default component weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        "entropy": 0.30,
        "trend": 0.15,
        "ml": 0.30,
        "masking": 0.10,
        "news2": 0.15,
    }

    # Severity thresholds on FRS (0-100)
    FRS_THRESHOLDS = {
        "watch": 26,
        "warning": 46,
        "critical": 71,
    }

    # Override constants
    CES_CRITICAL_THRESHOLD = 0.20
    ML_OVERRIDE_THRESHOLD = 0.80
    DRUG_MASKING_ML_THRESHOLD = 0.50
    ENTROPY_FLOOR_THRESHOLD = 0.30

    # Disagreement detection
    DISAGREEMENT_ENTROPY_RISK_THRESHOLD = 0.50
    DISAGREEMENT_ML_RISK_THRESHOLD = 0.30

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize with optional config overrides.

        Parameters
        ----------
        config : dict, optional
            Keys: 'weight_entropy', 'weight_trend', 'weight_ml',
                  'weight_masking', 'weight_news2', 'frs_thresholds'.
        """
        if config:
            self.weights = {
                "entropy": config.get("weight_entropy", self.DEFAULT_WEIGHTS["entropy"]),
                "trend": config.get("weight_trend", self.DEFAULT_WEIGHTS["trend"]),
                "ml": config.get("weight_ml", self.DEFAULT_WEIGHTS["ml"]),
                "masking": config.get("weight_masking", self.DEFAULT_WEIGHTS["masking"]),
                "news2": config.get("weight_news2", self.DEFAULT_WEIGHTS["news2"]),
            }
            thresholds = config.get("frs_thresholds", {})
            self.FRS_THRESHOLDS = {
                "watch": thresholds.get("watch", 26),
                "warning": thresholds.get("warning", 46),
                "critical": thresholds.get("critical", 71),
            }
        else:
            self.weights = dict(self.DEFAULT_WEIGHTS)

    def fuse(
        self,
        ces_adjusted: float,
        ces_slope_6h: float,
        ml_risk_1h: Optional[float],
        ml_risk_4h: Optional[float],
        ml_risk_8h: Optional[float],
        drug_masking: bool,
        news2_score: int,
    ) -> dict:
        """
        Fuse all risk signals into a single Final Risk Score.

        Parameters
        ----------
        ces_adjusted : float
            Drug-adjusted Composite Entropy Score (0-1, lower = worse).
        ces_slope_6h : float
            6-hour CES slope (negative = declining).
        ml_risk_1h : float or None
            ML deterioration risk at 1h horizon (0-1), None if unavailable.
        ml_risk_4h : float or None
            ML deterioration risk at 4h horizon (0-1), None if unavailable.
        ml_risk_8h : float or None
            ML deterioration risk at 8h horizon (0-1), None if unavailable.
        drug_masking : bool
            Whether drugs are currently masking vital sign changes.
        news2_score : int
            NEWS2 clinical score (0-20).

        Returns
        -------
        dict
            Complete fusion result with FRS, severity, time estimate,
            component risks, override info, and disagreement detection.
        """
        # Bug Fix #2: Initialize override_reason at top
        override_reason = None

        # Determine ML availability
        ml_available = ml_risk_4h is not None

        # ──────────────────────────────────────────
        # Step 1: Compute component risk scores (each 0-1)
        # ──────────────────────────────────────────

        # Entropy risk: CES is 0-1 where LOW = bad, so risk = 1 - CES
        entropy_risk = max(0.0, min(1.0, 1.0 - ces_adjusted))

        # Trend risk: negative slope = increasing risk
        # Scale: slope of -0.005/min is very bad → risk ≈ 1.0
        trend_risk = max(0.0, min(1.0, -ces_slope_6h / 0.005)) if ces_slope_6h < 0 else 0.0

        # ML risk: use the 4h horizon as primary (it's the balanced horizon)
        ml_risk = float(ml_risk_4h) if ml_available else 0.0

        # Drug masking risk: binary boost when masking is active
        masking_risk = 0.0
        if drug_masking:
            masking_risk = 0.30  # Base masking risk
            # Amplify if entropy is also declining
            if entropy_risk > 0.40:
                masking_risk = 0.50
            # Further amplify if ML agrees
            if ml_available and ml_risk > 0.50:
                masking_risk = 0.70

        # NEWS2 risk: scale 0-20 → 0-1, with threshold at 5
        news2_risk = max(0.0, min(1.0, news2_score / 12.0))

        # ──────────────────────────────────────────
        # Step 2: Weight redistribution if ML unavailable
        # ──────────────────────────────────────────

        weights = dict(self.weights)

        if not ml_available:
            # Redistribute ML weight to entropy (70%) and NEWS2 (30%)
            ml_weight = weights["ml"]
            weights["ml"] = 0.0
            weights["entropy"] += ml_weight * 0.70
            weights["news2"] += ml_weight * 0.30

        # ──────────────────────────────────────────
        # Step 3: Compute weighted Final Risk Score
        # ──────────────────────────────────────────

        component_risks = {
            "entropy": round(entropy_risk, 3),
            "trend": round(trend_risk, 3),
            "ml": round(ml_risk, 3),
            "masking": round(masking_risk, 3),
            "news2": round(news2_risk, 3),
        }

        weighted_sum = (
            weights["entropy"] * entropy_risk
            + weights["trend"] * trend_risk
            + weights["ml"] * ml_risk
            + weights["masking"] * masking_risk
            + weights["news2"] * news2_risk
        )

        frs = int(round(weighted_sum * 100))
        frs = max(0, min(100, frs))

        # ──────────────────────────────────────────
        # Step 4: Determine severity tier
        # ──────────────────────────────────────────

        if frs >= self.FRS_THRESHOLDS["critical"]:
            severity = "CRITICAL"
        elif frs >= self.FRS_THRESHOLDS["warning"]:
            severity = "WARNING"
        elif frs >= self.FRS_THRESHOLDS["watch"]:
            severity = "WATCH"
        else:
            severity = "NONE"

        # ──────────────────────────────────────────
        # Step 5: Override rules (safety nets)
        # ──────────────────────────────────────────

        # Override 1: Entropy CRITICAL always escalates
        if ces_adjusted < self.CES_CRITICAL_THRESHOLD:
            if severity != "CRITICAL":
                severity = "CRITICAL"
                frs = max(frs, self.FRS_THRESHOLDS["critical"])
                override_reason = "Entropy CRITICAL override (CES < 0.20)"

        # Override 2: ML predicts very high risk at 4h but severity is low
        if ml_available and ml_risk_4h > self.ML_OVERRIDE_THRESHOLD:
            if severity in ("NONE", "WATCH"):
                severity = "WARNING"
                frs = max(frs, self.FRS_THRESHOLDS["warning"])
                override_reason = "ML high-risk override (4h risk > 0.80)"

        # Override 3: Drug masking + ML elevated → floor at WARNING
        if drug_masking and ml_available and ml_risk_4h > self.DRUG_MASKING_ML_THRESHOLD:
            if severity in ("NONE", "WATCH"):
                severity = "WARNING"
                frs = max(frs, self.FRS_THRESHOLDS["warning"])
                override_reason = "Drug masking + ML elevated override"

        # Override 4: Entropy floor — CES < 0.30 should be at least WATCH
        if ces_adjusted < self.ENTROPY_FLOOR_THRESHOLD:
            if severity == "NONE":
                severity = "WATCH"
                frs = max(frs, self.FRS_THRESHOLDS["watch"])
                override_reason = "Entropy floor override (CES < 0.30)"

        # ──────────────────────────────────────────
        # Step 6: Disagreement detection
        # ──────────────────────────────────────────

        disagreement = self._detect_disagreement(
            entropy_risk, ml_risk, ml_available
        )

        # ──────────────────────────────────────────
        # Step 7: Time-to-event estimation
        # ──────────────────────────────────────────

        time_estimate = self._estimate_time_to_event(
            ml_risk_1h, ml_risk_4h, ml_risk_8h, ml_available, severity
        )

        return {
            "final_risk_score": frs,
            "final_severity": severity,
            "time_to_event_estimate": time_estimate,
            "component_risks": component_risks,
            "ml_available": ml_available,
            "override_applied": override_reason,
            "disagreement": disagreement,
        }

    def _detect_disagreement(
        self,
        entropy_risk: float,
        ml_risk: float,
        ml_available: bool,
    ) -> Optional[dict]:
        """
        Detect significant disagreement between entropy and ML assessments.

        Two types:
          - entropy_high_ml_low: Entropy says danger, ML says safe
          - entropy_low_ml_high: Entropy says safe, ML says danger
        """
        if not ml_available:
            return None

        # Entropy alarming, ML calm
        if (
            entropy_risk > self.DISAGREEMENT_ENTROPY_RISK_THRESHOLD
            and ml_risk < self.DISAGREEMENT_ML_RISK_THRESHOLD
        ):
            return {
                "type": "entropy_high_ml_low",
                "entropy_risk": round(entropy_risk, 3),
                "ml_risk": round(ml_risk, 3),
                "message": (
                    "Entropy patterns suggest elevated risk, but ML model "
                    "assesses low probability of deterioration."
                ),
                "resolution": "Conservative: entropy assessment maintained",
            }

        # ML alarming, entropy calm
        if (
            ml_risk > self.DISAGREEMENT_ENTROPY_RISK_THRESHOLD
            and entropy_risk < self.DISAGREEMENT_ML_RISK_THRESHOLD
        ):
            return {
                "type": "entropy_low_ml_high",
                "entropy_risk": round(entropy_risk, 3),
                "ml_risk": round(ml_risk, 3),
                "message": (
                    "ML model predicts elevated deterioration risk, but "
                    "entropy patterns appear relatively normal."
                ),
                "resolution": "Elevated monitoring recommended",
            }

        return None

    def _estimate_time_to_event(
        self,
        ml_risk_1h: Optional[float],
        ml_risk_4h: Optional[float],
        ml_risk_8h: Optional[float],
        ml_available: bool,
        severity: str,
    ) -> str:
        """
        Estimate time to potential deterioration event.

        Uses the risk gradient across time horizons to infer when
        risk is likely to cross the critical threshold.
        """
        if not ml_available:
            if severity == "CRITICAL":
                return "Imminent (entropy-based)"
            elif severity == "WARNING":
                return "Unknown (ML unavailable)"
            else:
                return "No imminent risk detected"

        # Use the risk values to infer timing
        r1 = ml_risk_1h if ml_risk_1h is not None else 0.0
        r4 = ml_risk_4h if ml_risk_4h is not None else 0.0
        r8 = ml_risk_8h if ml_risk_8h is not None else 0.0

        # If 1h risk is already high, event is imminent
        if r1 > 0.70:
            return "< 1 hour"
        elif r1 > 0.50:
            return "~1-2 hours"

        # If 4h risk is high but 1h is not, event is in 2-4h range
        if r4 > 0.70:
            return "~2-4 hours"
        elif r4 > 0.50:
            return "~4 hours"

        # If 8h risk is high but 4h is not, event is more distant
        if r8 > 0.70:
            return "~4-8 hours"
        elif r8 > 0.50:
            return "~8 hours"

        # Check for accelerating risk (steepening gradient)
        if r4 > r1 * 1.5 and r8 > r4 * 1.5:
            return "Accelerating risk pattern detected"

        return "No imminent risk detected"
