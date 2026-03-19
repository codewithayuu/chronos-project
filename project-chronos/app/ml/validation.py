"""
Model Compatibility Validation — ML Runtime for Project Chronos.

Validates that Person 1's regenerated model files are compatible with
Person 2's runtime classes. Checks:
  - Feature name/order consistency
  - Scaler dimensionality (n_features_in_ == 45)
  - Confidence parameter shapes
  - Syndrome classifier class count
  - Temperature scaler interface compliance
  - Warmup/sparsity assumptions

Run this after Person 1 delivers model files to data/models/.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple

import joblib

from .predictor import FEATURE_NAMES

# Expected constants
EXPECTED_NUM_FEATURES = 45
EXPECTED_NUM_SYNDROMES = 5
EXPECTED_SYNDROME_NAMES = [
    "sepsis_like",
    "respiratory_failure",
    "hemodynamic_instability",
    "cardiac_instability",
    "stable",
]

# Entropy feature indices (18-27) — these get imputed during warmup
ENTROPY_FEATURE_INDICES = list(range(18, 28))
ENTROPY_FEATURE_NAMES = [FEATURE_NAMES[i] for i in ENTROPY_FEATURE_INDICES]


class ModelValidationResult:
    """Result of a single validation check."""

    def __init__(self, check_name: str, passed: bool, message: str, severity: str = "INFO"):
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.severity = severity  # "INFO", "WARNING", "ERROR"

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.check_name}: {self.message}"


class ModelValidator:
    """
    Validates compatibility of Person 1's model artifacts with Person 2's runtime.

    Usage:
        validator = ModelValidator("data/models")
        results = validator.run_all()
        validator.print_report(results)
    """

    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)

    def run_all(self) -> List[ModelValidationResult]:
        """Run all validation checks and return results."""
        results = []

        # Check 1: Directory exists
        results.append(self._check_directory_exists())
        if not results[-1].passed:
            return results  # Can't continue without directory

        # Check 2: All required files exist
        results.extend(self._check_required_files())

        # Check 3: Scaler compatibility
        results.extend(self._check_scaler())

        # Check 4: Deterioration models
        results.extend(self._check_deterioration_models())

        # Check 5: Syndrome classifier
        results.extend(self._check_syndrome_classifier())

        # Check 6: Temperature scaler
        results.extend(self._check_temperature_scaler())

        # Check 7: Confidence parameters
        results.extend(self._check_confidence_params())

        # Check 8: Feature compatibility with synthetic test input
        results.extend(self._check_end_to_end_prediction())

        # Check 9: Warmup behavior (sparsity handling)
        results.extend(self._check_warmup_handling())

        return results

    def _check_directory_exists(self) -> ModelValidationResult:
        """Check that models directory exists."""
        if self.models_dir.exists():
            return ModelValidationResult(
                "models_dir", True,
                f"Models directory found: {self.models_dir}"
            )
        return ModelValidationResult(
            "models_dir", False,
            f"Models directory not found: {self.models_dir}. "
            "Person 1 has not delivered model files yet.",
            severity="ERROR"
        )

    def _check_required_files(self) -> List[ModelValidationResult]:
        """Check that all required model files exist."""
        required = [
            "deterioration_1h.joblib",
            "deterioration_4h.joblib",
            "deterioration_8h.joblib",
            "syndrome_classifier.joblib",
            "temperature_scaler.joblib",
            "scaler.joblib",
            "confidence_params.joblib",
        ]
        results = []
        for fname in required:
            path = self.models_dir / fname
            if path.exists():
                results.append(ModelValidationResult(
                    f"file_{fname}", True,
                    f"Found: {fname}"
                ))
            else:
                results.append(ModelValidationResult(
                    f"file_{fname}", False,
                    f"Missing: {fname}",
                    severity="ERROR"
                ))
        return results

    def _check_scaler(self) -> List[ModelValidationResult]:
        """Validate the StandardScaler dimensions and behavior."""
        results = []
        scaler_path = self.models_dir / "scaler.joblib"

        if not scaler_path.exists():
            return [ModelValidationResult(
                "scaler_check", False, "scaler.joblib not found", severity="ERROR"
            )]

        try:
            scaler = joblib.load(scaler_path)

            # Check n_features_in_
            if hasattr(scaler, "n_features_in_"):
                n_features = scaler.n_features_in_
                if n_features == EXPECTED_NUM_FEATURES:
                    results.append(ModelValidationResult(
                        "scaler_dimensions", True,
                        f"Scaler expects {n_features} features (correct)"
                    ))
                else:
                    results.append(ModelValidationResult(
                        "scaler_dimensions", False,
                        f"Scaler expects {n_features} features, but runtime expects {EXPECTED_NUM_FEATURES}. "
                        "Feature engineer output may have changed.",
                        severity="ERROR"
                    ))
            else:
                results.append(ModelValidationResult(
                    "scaler_dimensions", False,
                    "Scaler missing n_features_in_ attribute (may not be fitted)",
                    severity="WARNING"
                ))

            # Check that scaler can transform a 45-d vector without error
            try:
                test_input = np.zeros((1, EXPECTED_NUM_FEATURES))
                output = scaler.transform(test_input)
                if output.shape == (1, EXPECTED_NUM_FEATURES):
                    results.append(ModelValidationResult(
                        "scaler_transform", True,
                        "Scaler transforms (1, 45) -> (1, 45) correctly"
                    ))
                else:
                    results.append(ModelValidationResult(
                        "scaler_transform", False,
                        f"Scaler output shape: {output.shape}, expected (1, 45)",
                        severity="ERROR"
                    ))
            except Exception as e:
                results.append(ModelValidationResult(
                    "scaler_transform", False,
                    f"Scaler transform failed: {e}",
                    severity="ERROR"
                ))

            # Check for NaN in scaler mean/scale
            if hasattr(scaler, "mean_") and scaler.mean_ is not None:
                nan_count = np.isnan(scaler.mean_).sum()
                if nan_count > 0:
                    results.append(ModelValidationResult(
                        "scaler_nan_mean", False,
                        f"Scaler has {nan_count} NaN values in mean_ (sparsity issue from training data)",
                        severity="WARNING"
                    ))
            if hasattr(scaler, "scale_") and scaler.scale_ is not None:
                zero_count = (scaler.scale_ < 1e-10).sum()
                if zero_count > 0:
                    results.append(ModelValidationResult(
                        "scaler_zero_scale", False,
                        f"Scaler has {zero_count} near-zero scale values "
                        "(constant features in training data — may cause div-by-zero)",
                        severity="WARNING"
                    ))

        except Exception as e:
            results.append(ModelValidationResult(
                "scaler_load", False, f"Failed to load scaler: {e}", severity="ERROR"
            ))

        return results

    def _check_deterioration_models(self) -> List[ModelValidationResult]:
        """Validate the three deterioration models."""
        results = []

        for horizon in ["1h", "4h", "8h"]:
            fname = f"deterioration_{horizon}.joblib"
            path = self.models_dir / fname

            if not path.exists():
                results.append(ModelValidationResult(
                    f"det_{horizon}", False, f"{fname} not found", severity="ERROR"
                ))
                continue

            try:
                model = joblib.load(path)

                # Check predict_proba exists
                if hasattr(model, "predict_proba"):
                    results.append(ModelValidationResult(
                        f"det_{horizon}_interface", True,
                        f"{fname}: has predict_proba"
                    ))
                else:
                    results.append(ModelValidationResult(
                        f"det_{horizon}_interface", False,
                        f"{fname}: missing predict_proba method",
                        severity="ERROR"
                    ))

                # Check it's a CalibratedClassifierCV (as expected)
                if hasattr(model, "calibrated_classifiers_"):
                    results.append(ModelValidationResult(
                        f"det_{horizon}_calibrated", True,
                        f"{fname}: CalibratedClassifierCV detected"
                    ))

                    # Check the inner estimator is XGBoost
                    inner = model.calibrated_classifiers_[0].estimator
                    class_name = type(inner).__name__
                    if "XGB" in class_name:
                        results.append(ModelValidationResult(
                            f"det_{horizon}_xgb", True,
                            f"{fname}: inner estimator is {class_name}"
                        ))

                        # Check feature names match if available
                        if hasattr(inner, "get_booster"):
                            booster = inner.get_booster()
                            booster_features = booster.feature_names
                            if booster_features is not None:
                                if list(booster_features) == FEATURE_NAMES:
                                    results.append(ModelValidationResult(
                                        f"det_{horizon}_features", True,
                                        f"{fname}: feature names match runtime registry"
                                    ))
                                else:
                                    # Find mismatches
                                    mismatches: List[str] = []
                                    for i, (got, expected) in enumerate(
                                        zip(booster_features, FEATURE_NAMES)
                                    ):
                                        if got != expected:
                                            if len(mismatches) < 5:
                                                mismatches.append(f"  idx {i}: model='{got}' vs runtime='{expected}'")
                                    results.append(ModelValidationResult(
                                        f"det_{horizon}_features", False,
                                        f"{fname}: feature name mismatch!\n" + "\n".join(mismatches),
                                        severity="ERROR"
                                    ))
                    else:
                        results.append(ModelValidationResult(
                            f"det_{horizon}_xgb", False,
                            f"{fname}: inner estimator is {class_name}, expected XGBClassifier",
                            severity="WARNING"
                        ))
                else:
                    results.append(ModelValidationResult(
                        f"det_{horizon}_calibrated", False,
                        f"{fname}: not a CalibratedClassifierCV (predictions may be uncalibrated)",
                        severity="WARNING"
                    ))

            except Exception as e:
                results.append(ModelValidationResult(
                    f"det_{horizon}_load", False, f"Failed to load {fname}: {e}", severity="ERROR"
                ))

        return results

    def _check_syndrome_classifier(self) -> List[ModelValidationResult]:
        """Validate the syndrome classifier."""
        results = []
        path = self.models_dir / "syndrome_classifier.joblib"

        if not path.exists():
            return [ModelValidationResult(
                "syndrome_classifier", False, "syndrome_classifier.joblib not found", severity="ERROR"
            )]

        try:
            model = joblib.load(path)

            # Check predict_proba
            if hasattr(model, "predict_proba"):
                results.append(ModelValidationResult(
                    "syndrome_interface", True, "syndrome_classifier has predict_proba"
                ))
            else:
                results.append(ModelValidationResult(
                    "syndrome_interface", False, "Missing predict_proba", severity="ERROR"
                ))

            # Check number of classes
            if hasattr(model, "n_classes_"):
                n_classes = model.n_classes_
                if n_classes == EXPECTED_NUM_SYNDROMES:
                    results.append(ModelValidationResult(
                        "syndrome_classes", True,
                        f"Syndrome classifier has {n_classes} classes (correct)"
                    ))
                else:
                    results.append(ModelValidationResult(
                        "syndrome_classes", False,
                        f"Syndrome classifier has {n_classes} classes, expected {EXPECTED_NUM_SYNDROMES}",
                        severity="ERROR"
                    ))
            elif hasattr(model, "classes_"):
                n_classes = len(model.classes_)
                if n_classes == EXPECTED_NUM_SYNDROMES:
                    results.append(ModelValidationResult(
                        "syndrome_classes", True,
                        f"Syndrome classifier has {n_classes} classes (correct)"
                    ))
                else:
                    results.append(ModelValidationResult(
                        "syndrome_classes", False,
                        f"Syndrome classifier has {n_classes} classes, expected {EXPECTED_NUM_SYNDROMES}",
                        severity="ERROR"
                    ))

        except Exception as e:
            results.append(ModelValidationResult(
                "syndrome_load", False, f"Failed to load syndrome classifier: {e}", severity="ERROR"
            ))

        return results

    def _check_temperature_scaler(self) -> List[ModelValidationResult]:
        """Validate the temperature scaler interface."""
        results = []
        path = self.models_dir / "temperature_scaler.joblib"

        if not path.exists():
            return [ModelValidationResult(
                "temp_scaler", False, "temperature_scaler.joblib not found", severity="ERROR"
            )]

        try:
            temp_scaler = joblib.load(path)

            # Check calibrate method exists
            if hasattr(temp_scaler, "calibrate"):
                results.append(ModelValidationResult(
                    "temp_scaler_interface", True,
                    "TemperatureScaler has .calibrate() method"
                ))

                # Test with dummy probabilities
                try:
                    dummy_probs = np.array([[0.3, 0.2, 0.2, 0.2, 0.1]])
                    output = temp_scaler.calibrate(dummy_probs)
                    if output.shape == (1, EXPECTED_NUM_SYNDROMES):
                        results.append(ModelValidationResult(
                            "temp_scaler_output", True,
                            f"calibrate() output shape (1, {EXPECTED_NUM_SYNDROMES}) correct"
                        ))
                    else:
                        results.append(ModelValidationResult(
                            "temp_scaler_output", False,
                            f"calibrate() output shape {output.shape}, expected (1, {EXPECTED_NUM_SYNDROMES})",
                            severity="ERROR"
                        ))

                    # Verify output sums to ~1
                    prob_sum = float(output.sum())
                    if 0.9 < prob_sum < 1.1:
                        results.append(ModelValidationResult(
                            "temp_scaler_sum", True,
                            f"Calibrated probabilities sum to {prob_sum:.3f} (close to 1.0)"
                        ))
                    else:
                        results.append(ModelValidationResult(
                            "temp_scaler_sum", False,
                            f"Calibrated probabilities sum to {prob_sum:.3f} (should be ~1.0)",
                            severity="WARNING"
                        ))
                except Exception as e:
                    results.append(ModelValidationResult(
                        "temp_scaler_test", False,
                        f"calibrate() failed with test input: {e}",
                        severity="ERROR"
                    ))
            else:
                results.append(ModelValidationResult(
                    "temp_scaler_interface", False,
                    "TemperatureScaler missing .calibrate() method",
                    severity="ERROR"
                ))

        except Exception as e:
            results.append(ModelValidationResult(
                "temp_scaler_load", False,
                f"Failed to load temperature scaler: {e}",
                severity="ERROR"
            ))

        return results

    def _check_confidence_params(self) -> List[ModelValidationResult]:
        """Validate the confidence parameters (centroid + cov_inv)."""
        results = []
        path = self.models_dir / "confidence_params.joblib"

        if not path.exists():
            return [ModelValidationResult(
                "confidence_params", False,
                "confidence_params.joblib not found",
                severity="ERROR"
            )]

        try:
            conf = joblib.load(path)

            if not isinstance(conf, dict):
                return [ModelValidationResult(
                    "confidence_params_type", False,
                    f"Expected dict, got {type(conf).__name__}",
                    severity="ERROR"
                )]

            # Check centroid
            if "centroid" in conf:
                centroid = conf["centroid"]
                if centroid.shape == (EXPECTED_NUM_FEATURES,):
                    results.append(ModelValidationResult(
                        "centroid_shape", True,
                        f"Centroid shape ({EXPECTED_NUM_FEATURES},) correct"
                    ))
                else:
                    results.append(ModelValidationResult(
                        "centroid_shape", False,
                        f"Centroid shape {centroid.shape}, expected ({EXPECTED_NUM_FEATURES},)",
                        severity="ERROR"
                    ))
            else:
                results.append(ModelValidationResult(
                    "centroid_missing", False, "Missing 'centroid' key", severity="ERROR"
                ))

            # Check cov_inv
            if "cov_inv" in conf:
                cov_inv = conf["cov_inv"]
                expected_shape = (EXPECTED_NUM_FEATURES, EXPECTED_NUM_FEATURES)
                if cov_inv.shape == expected_shape:
                    results.append(ModelValidationResult(
                        "cov_inv_shape", True,
                        f"Covariance inverse shape {expected_shape} correct"
                    ))

                    # Check for NaN/Inf (pseudo-inverse may produce these)
                    nan_count = np.isnan(cov_inv).sum()
                    inf_count = np.isinf(cov_inv).sum()
                    if nan_count > 0 or inf_count > 0:
                        results.append(ModelValidationResult(
                            "cov_inv_numeric", False,
                            f"Covariance inverse has {nan_count} NaN and {inf_count} Inf values",
                            severity="WARNING"
                        ))
                else:
                    results.append(ModelValidationResult(
                        "cov_inv_shape", False,
                        f"Covariance inverse shape {cov_inv.shape}, expected {expected_shape}",
                        severity="ERROR"
                    ))
            else:
                results.append(ModelValidationResult(
                    "cov_inv_missing", False, "Missing 'cov_inv' key", severity="ERROR"
                ))

        except Exception as e:
            results.append(ModelValidationResult(
                "confidence_load", False,
                f"Failed to load confidence params: {e}",
                severity="ERROR"
            ))

        return results

    def _check_end_to_end_prediction(self) -> List[ModelValidationResult]:
        """Run a full prediction pipeline with synthetic test data."""
        results = []

        try:
            from .predictor import DeteriorationPredictor
            from .classifier import SyndromeClassifier

            # Create predictor and classifier
            predictor = DeteriorationPredictor(str(self.models_dir))
            classifier = SyndromeClassifier(str(self.models_dir))

            if not predictor.available:
                results.append(ModelValidationResult(
                    "e2e_predictor", False,
                    "DeteriorationPredictor not available (models missing)",
                    severity="WARNING"
                ))
                return results

            # Generate a realistic test vector (not random — use plausible vitals)
            test_features = np.array([
                80.0,   # hr_current
                120.0,  # bp_sys_current
                75.0,   # bp_dia_current
                16.0,   # rr_current
                97.0,   # spo2_current
                37.0,   # temp_current
                78.0, 8.0, 118.0, 10.0,  # hr/bp 6h stats
                15.0, 2.0, 96.5, 1.5,   # rr/spo2 6h stats
                37.1, 0.3, 65.0, 95.0,  # temp 6h stats, hr min/max
                1.2, 1.0, 0.9, 0.8,     # sampen_hr, bp, rr, spo2
                0.65, 0.60, 0.0, 0.0,   # ces_raw, ces_adj, slopes
                0.0, 0.1,               # mse_slope, entropy_divergence
                0.67, 0.96, 45.0,       # shock_index, mod_shock, pulse_pressure
                90.0, 0.165, 5.0,       # map, rr_spo2, hr_rr
                0.0, 0.0, 0.0, 0.0,    # drug features
                0.0, 0.0, 0.0, 0.0,    # trend slopes
                0.0, 3.0, 0.6,         # divergence_x_ces, news2, age_norm
            ])

            det_result = predictor.predict(test_features)
            if det_result is not None:
                # Validate output format
                required_keys = ["risk_1h", "risk_4h", "risk_8h", "model_confidence", "top_drivers"]
                missing = [k for k in required_keys if k not in det_result]
                if not missing:
                    results.append(ModelValidationResult(
                        "e2e_det_prediction", True,
                        f"Prediction successful: risk_4h={det_result['risk_4h']:.3f}, "
                        f"confidence={det_result['model_confidence']}"
                    ))
                else:
                    results.append(ModelValidationResult(
                        "e2e_det_prediction", False,
                        f"Missing keys in prediction output: {missing}",
                        severity="ERROR"
                    ))

                # Check risk values are in [0, 1]
                for key in ["risk_1h", "risk_4h", "risk_8h"]:
                    val = det_result.get(key, -1)
                    if not (0.0 <= val <= 1.0):
                        results.append(ModelValidationResult(
                            f"e2e_{key}_range", False,
                            f"{key} = {val}, expected [0, 1]",
                            severity="ERROR"
                        ))
            else:
                results.append(ModelValidationResult(
                    "e2e_det_prediction", False,
                    "Prediction returned None despite models being available",
                    severity="ERROR"
                ))

            # Test syndrome classifier
            if classifier.available:
                syn_result = classifier.predict(test_features)
                if syn_result is not None:
                    results.append(ModelValidationResult(
                        "e2e_syn_prediction", True,
                        f"Syndrome prediction: {syn_result['primary_syndrome']} "
                        f"({syn_result['primary_confidence']:.2f})"
                    ))
                else:
                    results.append(ModelValidationResult(
                        "e2e_syn_prediction", False,
                        "Syndrome prediction returned None",
                        severity="ERROR"
                    ))

        except Exception as e:
            results.append(ModelValidationResult(
                "e2e_test", False,
                f"End-to-end test failed: {e}",
                severity="ERROR"
            ))

        return results

    def _check_warmup_handling(self) -> List[ModelValidationResult]:
        """
        Validate that models handle warmup-imputed features correctly.

        During warmup, entropy features (indices 18-27) are imputed with
        population medians. Verify models don't crash or produce NaN 
        with these imputed values.
        """
        results = []

        try:
            from .predictor import DeteriorationPredictor

            predictor = DeteriorationPredictor(str(self.models_dir))
            if not predictor.available:
                return [ModelValidationResult(
                    "warmup_test", False,
                    "Predictor not available, cannot test warmup handling",
                    severity="WARNING"
                )]

            # Create feature vector with entropy features set to population medians
            warmup_features = np.zeros(EXPECTED_NUM_FEATURES)
            # Set non-entropy features to plausible values
            warmup_features[0] = 80.0   # hr
            warmup_features[1] = 120.0  # bp_sys
            warmup_features[2] = 75.0   # bp_dia
            warmup_features[3] = 16.0   # rr
            warmup_features[4] = 97.0   # spo2
            warmup_features[5] = 37.0   # temp
            # Entropy features default to 0 (as impute_warmup would set them to medians)

            result = predictor.predict(warmup_features)
            if result is not None:
                # Check for NaN in output
                has_nan = any(
                    np.isnan(result[k]) if isinstance(result[k], float) else False
                    for k in ["risk_1h", "risk_4h", "risk_8h"]
                )
                if not has_nan:
                    results.append(ModelValidationResult(
                        "warmup_prediction", True,
                        "Models handle warmup-imputed features without NaN"
                    ))
                else:
                    results.append(ModelValidationResult(
                        "warmup_prediction", False,
                        "Models produce NaN with warmup-imputed features",
                        severity="ERROR"
                    ))
            else:
                results.append(ModelValidationResult(
                    "warmup_prediction", False,
                    "Prediction returned None for warmup features",
                    severity="ERROR"
                ))

            # Test with ALL zeros (extreme sparsity edge case)
            all_zeros = np.zeros(EXPECTED_NUM_FEATURES)
            result = predictor.predict(all_zeros)
            if result is not None:
                has_nan = any(
                    np.isnan(result[k]) if isinstance(result[k], float) else False
                    for k in ["risk_1h", "risk_4h", "risk_8h"]
                )
                if not has_nan:
                    results.append(ModelValidationResult(
                        "sparsity_zeros", True,
                        "Models handle all-zero input without NaN"
                    ))
                else:
                    results.append(ModelValidationResult(
                        "sparsity_zeros", False,
                        "Models produce NaN with all-zero input (extreme sparsity)",
                        severity="WARNING"
                    ))

            # Test with NaN inputs (runtime handles nan_to_num, but verify)
            nan_features = np.full(EXPECTED_NUM_FEATURES, np.nan)
            result = predictor.predict(nan_features)
            if result is not None:
                results.append(ModelValidationResult(
                    "sparsity_nan", True,
                    "Models handle NaN input (converted to 0 by runtime)"
                ))
            # None result is also acceptable — nan_to_num converts to zero first

        except Exception as e:
            results.append(ModelValidationResult(
                "warmup_test", False,
                f"Warmup handling test failed: {e}",
                severity="ERROR"
            ))

        return results

    def print_report(self, results: List[ModelValidationResult]):
        """Print a formatted validation report."""
        print("\n" + "=" * 60)
        print("  MODEL COMPATIBILITY VALIDATION REPORT")
        print("=" * 60)

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        errors = sum(1 for r in results if not r.passed and r.severity == "ERROR")
        warnings = sum(1 for r in results if not r.passed and r.severity == "WARNING")

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            icon = "  " if r.passed else "!!"
            print(f"  {icon} [{status}] {r.check_name}: {r.message}")

        print("\n" + "-" * 60)
        print(f"  Total: {len(results)} | Passed: {passed} | "
              f"Failed: {failed} (Errors: {errors}, Warnings: {warnings})")
        print("=" * 60)

        if errors > 0:
            print("\n  ACTION REQUIRED: Fix ERROR-level issues before integration.")
        elif warnings > 0:
            print("\n  Some warnings detected. Review before demo.")
        else:
            print("\n  All checks passed. Models are compatible with runtime.")


def validate_models(models_dir: str = "data/models") -> bool:
    """
    Convenience function to validate models and print report.
    Returns True if no ERROR-level failures.
    """
    validator = ModelValidator(models_dir)
    results = validator.run_all()
    validator.print_report(results)
    errors = sum(1 for r in results if not r.passed and r.severity == "ERROR")
    return errors == 0


# Allow running directly: python -m app.ml.validation
if __name__ == "__main__":
    import sys
    models_dir = sys.argv[1] if len(sys.argv) > 1 else "data/models"
    success = validate_models(models_dir)
    sys.exit(0 if success else 1)
