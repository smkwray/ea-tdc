"""Retain raw covariance and qualify its symmetric covariance operator."""
from __future__ import annotations

import numpy as np

COVARIANCE_OPERATOR_POLICY = {
    "definition": "C=H(D_V_raw_D); raw covariance retained unchanged",
    "representation_relative_budget": 1e-7,
    "psd_relative_budget": 1e-7,
    "spectral_norm_floor": 1e-30,
    "reported_diagonal": "nonnegative_without_clipping",
    "treatment_variance": "strictly_positive",
}


def canonical_covariance(
    raw: object, scales: object, outcome_scale: float, *,
    treatment_index: int = 1, policy: dict | None = None,
) -> dict:
    """Return raw evidence, the symmetric operator, and qualified covariance.

    The original-unit covariance preserves raw diagonal values exactly. No
    eigenvalues or variances are clipped, and raw input arrays are never mutated.
    """
    if policy is not None and policy != COVARIANCE_OPERATOR_POLICY:
        raise ValueError("Covariance operator policy differs from the fixed definition")
    policy = COVARIANCE_OPERATOR_POLICY
    covariance = np.asarray(raw, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or scales.shape != (covariance.shape[0],) or covariance.size == 0
            or not 0 <= treatment_index < covariance.shape[0]
            or not np.isfinite(covariance).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0) or not np.isfinite(outcome_scale) or outcome_scale <= 0):
        raise ValueError("Invalid covariance dimensions, values, or normalization")
    diagonal = np.diag(covariance)
    if np.any(diagonal < 0):
        raise ValueError("Negative reported covariance diagonal")
    if diagonal[treatment_index] <= 0:
        raise ValueError("Treatment variance must be strictly positive")
    d = scales / outcome_scale
    conversion = np.outer(d, d)
    if not np.isfinite(conversion).all() or np.any(conversion <= 0):
        raise ValueError("Invalid scaled covariance conversion")
    scaled = covariance * conversion
    operator = (scaled + scaled.T) / 2
    if not np.isfinite(scaled).all() or not np.isfinite(operator).all():
        raise ValueError("Nonfinite scaled covariance operator")
    norm = max(float(np.linalg.norm(operator, 2)), policy["spectral_norm_floor"])
    defect = float(np.linalg.norm(scaled - operator, 2)) / norm
    if not np.isfinite(norm) or not np.isfinite(defect):
        raise ValueError("Nonfinite covariance norm or representation defect")
    if defect > policy["representation_relative_budget"]:
        raise ValueError("Raw covariance representation defect exceeds budget")
    minimum = float(np.linalg.eigvalsh(operator)[0])
    raw_skew = float(np.linalg.norm(scaled - scaled.T, 2)) / norm
    if not np.isfinite(minimum) or not np.isfinite(raw_skew):
        raise ValueError("Nonfinite covariance eigenvalue or raw skew")
    if minimum < -policy["psd_relative_budget"] * norm:
        raise ValueError("Canonical covariance operator is materially indefinite")
    canonical = operator / conversion
    np.fill_diagonal(canonical, diagonal)
    if not np.isfinite(canonical).all():
        raise ValueError("Nonfinite original-unit canonical covariance")
    return {
        "raw_covariance": covariance.tolist(), "covariance": canonical.tolist(),
        "scaled_raw_covariance": scaled.tolist(), "covariance_operator": operator.tolist(),
        "diagnostics": {"representation_relative_error": defect,
                        "raw_skew_relative_norm": raw_skew,
                        "minimum_scaled_eigenvalue": minimum, "covariance_operator_norm": norm},
    }
