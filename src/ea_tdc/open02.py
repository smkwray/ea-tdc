"""Fail-closed, in-memory implementation of the frozen OPEN-02 contract.

The module contains the statistical primitives and the live OPEN-02 pipeline.
It deliberately performs no network access or filesystem writes so a producer
can supply already-acquired, hashable inputs and retain the resulting evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import erfc, exp, inf, isfinite, lgamma, log, sqrt
from typing import Any, Mapping, Sequence

from ea_tdc.estimation import (
    _invert,
    _matmul,
    _matvec,
    _outer,
    _transpose,
    _zeros,
)
from ea_tdc.open_contract import (
    OPEN02_BANK_AGENCY_ID,
    OPEN02_BANK_DEPOSITS_ID,
    OPEN02_BANK_LOANS_ID,
    OPEN02_BANK_TREASURY_ID,
    OPEN02_CONTRACT,
    OPEN02_LEAVE_OUT_ID,
    Open02Contract,
    Open02CovarianceContract,
)


OPEN02_HAC_LAGS = OPEN02_CONTRACT.covariance.lag_quarters
OPEN02_HOLM_FAMILY_SIZE = len(OPEN02_CONTRACT.holm.hypothesis_ids)
OPEN02_INFLUENCE_DENOMINATOR_FLOOR = (
    OPEN02_CONTRACT.influence.relative_l2_denominator_floor
)
OPEN02_LEAVE_ONE_INFLUENCE_THRESHOLD = (
    OPEN02_CONTRACT.influence.maximum_quarter_influence
)
OPEN02_LEAVE_BLOCK_INFLUENCE_THRESHOLD = (
    OPEN02_CONTRACT.influence.maximum_block_influence
)
OPEN02_SIGNIFICANCE_ALPHA = (
    OPEN02_CONTRACT.influence.sign_stability_raw_p_threshold
)


@dataclass(frozen=True)
class OLSFit:
    """Full-rank OLS coefficients and observation-level residual evidence."""

    coefficients: tuple[float, ...]
    fitted_values: tuple[float, ...]
    residuals: tuple[float, ...]
    observations: int
    parameters: int


@dataclass(frozen=True)
class StackedSystemFit:
    """Identical-design equation fits and their stacked HAC covariance."""

    equation_fits: tuple[OLSFit, ...]
    covariance: tuple[tuple[float, ...], ...]
    observations: int
    parameters_per_equation: int
    hac_lags: int
    kernel: str
    prewhitened: bool
    finite_sample_scale: float

    @property
    def equation_count(self) -> int:
        return len(self.equation_fits)

    @property
    def flat_coefficients(self) -> tuple[float, ...]:
        return tuple(
            coefficient
            for fit in self.equation_fits
            for coefficient in fit.coefficients
        )

    def flat_index(self, equation_index: int, coefficient_index: int) -> int:
        if not 0 <= equation_index < self.equation_count:
            raise IndexError(f"Equation index out of range: {equation_index}")
        if not 0 <= coefficient_index < self.parameters_per_equation:
            raise IndexError(
                f"Coefficient index out of range: {coefficient_index}"
            )
        return (
            equation_index * self.parameters_per_equation
            + coefficient_index
        )


@dataclass(frozen=True)
class CoefficientSelection:
    """One declared equation/coefficient pair in a zero-restriction Wald test."""

    equation_index: int
    coefficient_index: int


@dataclass(frozen=True)
class WaldTestResult:
    selections: tuple[CoefficientSelection, ...]
    statistic: float
    degrees_of_freedom: int
    p_value: float


@dataclass(frozen=True)
class InfluenceGateResult:
    """Deterministic OPEN-02 influence and significant-sign stability result."""

    leave_one_influences: tuple[float, ...]
    leave_block_influences: tuple[float, ...]
    maximum_leave_one_influence: float
    maximum_leave_block_influence: float
    leave_one_passed: bool
    leave_block_passed: bool
    sign_flip_detected: bool
    passed: bool
    reason_codes: tuple[str, ...]


class Open02ValidationError(ValueError):
    """A deterministic OPEN-02 validity-gate failure."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


@dataclass(frozen=True)
class Open02PipelineResult:
    """All retained in-memory evidence produced by one OPEN-02 execution."""

    panel_rows: tuple[dict[str, Any], ...]
    estimate_rows: tuple[dict[str, Any], ...]
    wald_rows: tuple[dict[str, Any], ...]
    influence_rows: tuple[dict[str, Any], ...]
    influence_summaries: tuple[dict[str, Any], ...]
    acceptance: dict[str, Any]


def _finite_vector(
    values: Sequence[float],
    *,
    label: str,
    allow_empty: bool = False,
) -> list[float]:
    output = [float(value) for value in values]
    if not output and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if any(not isfinite(value) for value in output):
        raise ValueError(f"{label} contains a non-finite value")
    return output


def _validated_design(
    x_rows: Sequence[Sequence[float]],
) -> list[list[float]]:
    rows = [
        _finite_vector(row, label=f"Design row {index}")
        for index, row in enumerate(x_rows)
    ]
    if not rows:
        raise ValueError("Design matrix must not be empty")
    parameters = len(rows[0])
    if parameters == 0:
        raise ValueError("Design matrix must have at least one column")
    if any(len(row) != parameters for row in rows):
        raise ValueError("Design matrix rows have inconsistent dimensions")
    if len(rows) < parameters:
        raise ValueError(
            "Design matrix cannot be full rank when observations < parameters"
        )
    return rows


def _design_inverse(x_rows: list[list[float]]) -> list[list[float]]:
    cross_product = _matmul(_transpose(x_rows), x_rows)
    try:
        return _invert(cross_product)
    except ValueError as exc:
        raise ValueError("OPEN-02 design matrix is not full rank") from exc


def _fit_with_inverse(
    y_values: list[float],
    x_rows: list[list[float]],
    xtx_inverse: list[list[float]],
) -> OLSFit:
    coefficients = _matvec(
        xtx_inverse,
        _matvec(_transpose(x_rows), y_values),
    )
    fitted = [
        sum(
            coefficient * regressor
            for coefficient, regressor in zip(coefficients, row, strict=True)
        )
        for row in x_rows
    ]
    residuals = [
        actual - estimate
        for actual, estimate in zip(y_values, fitted, strict=True)
    ]
    return OLSFit(
        coefficients=tuple(coefficients),
        fitted_values=tuple(fitted),
        residuals=tuple(residuals),
        observations=len(x_rows),
        parameters=len(x_rows[0]),
    )


def fit_ols(
    y_values: Sequence[float],
    x_rows: Sequence[Sequence[float]],
) -> OLSFit:
    """Fit fail-closed, full-column-rank OLS and retain all residuals."""

    design = _validated_design(x_rows)
    outcome = _finite_vector(y_values, label="OLS outcome")
    if len(outcome) != len(design):
        raise ValueError(
            "OLS outcome and design must have identical observation counts"
        )
    return _fit_with_inverse(outcome, design, _design_inverse(design))


def _block_diagonal_bread(
    xtx_inverse: list[list[float]],
    equation_count: int,
) -> list[list[float]]:
    parameters = len(xtx_inverse)
    dimension = equation_count * parameters
    bread = _zeros(dimension, dimension)
    for equation in range(equation_count):
        offset = equation * parameters
        for row in range(parameters):
            for column in range(parameters):
                bread[offset + row][offset + column] = (
                    xtx_inverse[row][column]
                )
    return bread


def _symmetrized(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    output = _zeros(size, size)
    for row in range(size):
        for column in range(row, size):
            value = 0.5 * (matrix[row][column] + matrix[column][row])
            output[row][column] = value
            output[column][row] = value
    return output


def fit_stacked_system(
    outcomes_by_equation: Sequence[Sequence[float]],
    x_rows: Sequence[Sequence[float]],
    *,
    hac_lags: int = OPEN02_HAC_LAGS,
) -> StackedSystemFit:
    """Fit identical-X equations and their stacked Bartlett Newey-West HAC.

    Coefficients and covariance blocks use equation-major order.  Scores are
    ``x_t * residual[j, t]``.  No prewhitening is applied, and the entire HAC
    meat receives the frozen ``T / (T - K)`` finite-sample correction.
    """

    design = _validated_design(x_rows)
    observations = len(design)
    parameters = len(design[0])
    if observations <= parameters:
        raise ValueError(
            "Stacked HAC requires observations > parameters for T/(T-K)"
        )
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, int):
        raise TypeError("HAC lags must be an integer")
    if hac_lags < 0 or hac_lags >= observations:
        raise ValueError(
            "HAC lags must be nonnegative and smaller than observations"
        )
    if not outcomes_by_equation:
        raise ValueError("Stacked system must contain at least one equation")

    outcomes = [
        _finite_vector(values, label=f"Equation {index} outcome")
        for index, values in enumerate(outcomes_by_equation)
    ]
    if any(len(values) != observations for values in outcomes):
        raise ValueError(
            "Every stacked-system outcome must match the common design rows"
        )

    xtx_inverse = _design_inverse(design)
    equation_fits = tuple(
        _fit_with_inverse(values, design, xtx_inverse)
        for values in outcomes
    )
    equation_count = len(equation_fits)
    score_dimension = equation_count * parameters
    scores: list[list[float]] = []
    for observation, regressors in enumerate(design):
        score: list[float] = []
        for fit in equation_fits:
            residual = fit.residuals[observation]
            score.extend(regressor * residual for regressor in regressors)
        scores.append(score)

    meat = _zeros(score_dimension, score_dimension)
    for score in scores:
        addition = _outer(score, score)
        for row in range(score_dimension):
            for column in range(score_dimension):
                meat[row][column] += addition[row][column]
    for lag in range(1, hac_lags + 1):
        weight = 1.0 - lag / (hac_lags + 1.0)
        for observation in range(lag, observations):
            current = scores[observation]
            previous = scores[observation - lag]
            forward = _outer(current, previous)
            reverse = _outer(previous, current)
            for row in range(score_dimension):
                for column in range(score_dimension):
                    meat[row][column] += weight * (
                        forward[row][column] + reverse[row][column]
                    )

    finite_sample_scale = observations / (observations - parameters)
    for row in range(score_dimension):
        for column in range(score_dimension):
            meat[row][column] *= finite_sample_scale

    bread = _block_diagonal_bread(xtx_inverse, equation_count)
    covariance = _symmetrized(
        _matmul(_matmul(bread, meat), bread)
    )
    return StackedSystemFit(
        equation_fits=equation_fits,
        covariance=tuple(tuple(row) for row in covariance),
        observations=observations,
        parameters_per_equation=parameters,
        hac_lags=hac_lags,
        kernel="bartlett",
        prewhitened=False,
        finite_sample_scale=finite_sample_scale,
    )


def _regularized_gamma_q(shape: float, argument: float) -> float:
    """Regularized upper incomplete gamma using stable series/CF branches."""

    if shape <= 0.0 or argument < 0.0:
        raise ValueError("Gamma shape must be positive and argument nonnegative")
    if argument == 0.0:
        return 1.0
    epsilon = 1e-14
    maximum_iterations = 10000
    minimum = 1e-300
    log_prefactor = (
        -argument + shape * log(argument) - lgamma(shape)
    )

    if argument < shape + 1.0:
        term = 1.0 / shape
        total = term
        shifted_shape = shape
        for _ in range(maximum_iterations):
            shifted_shape += 1.0
            term *= argument / shifted_shape
            total += term
            if abs(term) <= abs(total) * epsilon:
                lower = total * exp(log_prefactor)
                return min(1.0, max(0.0, 1.0 - lower))
        raise ArithmeticError("Incomplete-gamma series did not converge")

    b_value = argument + 1.0 - shape
    c_value = 1.0 / minimum
    d_value = 1.0 / (
        b_value if abs(b_value) >= minimum else minimum
    )
    fraction = d_value
    for iteration in range(1, maximum_iterations + 1):
        coefficient = -iteration * (iteration - shape)
        b_value += 2.0
        d_value = coefficient * d_value + b_value
        if abs(d_value) < minimum:
            d_value = minimum
        c_value = b_value + coefficient / c_value
        if abs(c_value) < minimum:
            c_value = minimum
        d_value = 1.0 / d_value
        delta = d_value * c_value
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            upper = exp(log_prefactor) * fraction
            return min(1.0, max(0.0, upper))
    raise ArithmeticError("Incomplete-gamma continued fraction did not converge")


def chi_square_survival(
    statistic: float,
    degrees_of_freedom: int,
) -> float:
    """Return ``P[ChiSquare(df) >= statistic]`` without SciPy."""

    value = float(statistic)
    if isinstance(degrees_of_freedom, bool) or not isinstance(
        degrees_of_freedom, int
    ):
        raise TypeError("Chi-square degrees of freedom must be an integer")
    if degrees_of_freedom <= 0:
        raise ValueError("Chi-square degrees of freedom must be positive")
    if value < 0.0 or value != value:
        raise ValueError("Chi-square statistic must be nonnegative")
    if value == inf:
        return 0.0
    if not isfinite(value):
        raise ValueError("Chi-square statistic must be finite or positive infinity")
    return _regularized_gamma_q(
        degrees_of_freedom / 2.0,
        value / 2.0,
    )


def _scaled_inverse(matrix: list[list[float]], *, label: str) -> list[list[float]]:
    scale = max(abs(value) for row in matrix for value in row)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} is singular")
    normalized = [
        [value / scale for value in row]
        for row in matrix
    ]
    try:
        inverse = _invert(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} is singular") from exc
    return [
        [value / scale for value in row]
        for row in inverse
    ]


def wald_zero_test(
    system: StackedSystemFit,
    selections: Sequence[CoefficientSelection],
) -> WaldTestResult:
    """Test that the declared, unique system coefficients are jointly zero."""

    declared = tuple(selections)
    if not declared:
        raise ValueError("Wald test must select at least one coefficient")
    if len(set(declared)) != len(declared):
        raise ValueError("Wald coefficient selections must be unique")
    flat_indices = tuple(
        system.flat_index(
            selection.equation_index,
            selection.coefficient_index,
        )
        for selection in declared
    )
    coefficients = system.flat_coefficients
    selected_coefficients = [coefficients[index] for index in flat_indices]
    selected_covariance = [
        [system.covariance[row][column] for column in flat_indices]
        for row in flat_indices
    ]
    covariance_inverse = _scaled_inverse(
        selected_covariance,
        label="Selected Wald covariance",
    )
    statistic = sum(
        left * inverse * right
        for left, inverse_row in zip(
            selected_coefficients,
            covariance_inverse,
            strict=True,
        )
        for inverse, right in zip(
            inverse_row,
            selected_coefficients,
            strict=True,
        )
    )
    if statistic < -1e-10:
        raise ValueError("Wald statistic is negative; covariance is not valid")
    statistic = max(0.0, statistic)
    degrees_of_freedom = len(declared)
    return WaldTestResult(
        selections=declared,
        statistic=statistic,
        degrees_of_freedom=degrees_of_freedom,
        p_value=chi_square_survival(statistic, degrees_of_freedom),
    )


def holm_adjust_three(p_values: Sequence[float]) -> tuple[float, float, float]:
    """Holm-adjust exactly the three predeclared OPEN-02 Wald p-values."""

    raw = _finite_vector(p_values, label="Holm p-values")
    if len(raw) != OPEN02_HOLM_FAMILY_SIZE:
        raise ValueError("OPEN-02 Holm family must contain exactly three p-values")
    if any(value < 0.0 or value > 1.0 for value in raw):
        raise ValueError("Holm p-values must lie in [0, 1]")

    ordered = sorted(range(len(raw)), key=lambda index: (raw[index], index))
    adjusted = [0.0] * len(raw)
    running_maximum = 0.0
    for rank, original_index in enumerate(ordered):
        candidate = (len(raw) - rank) * raw[original_index]
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = min(1.0, running_maximum)
    return adjusted[0], adjusted[1], adjusted[2]


def relative_l2_influence(
    full_coefficients: Sequence[float],
    refit_coefficients: Sequence[float],
    *,
    denominator_floor: float = OPEN02_INFLUENCE_DENOMINATOR_FLOOR,
) -> float:
    """Compute the frozen relative L2 coefficient-change influence measure."""

    full = _finite_vector(full_coefficients, label="Full coefficients")
    refit = _finite_vector(refit_coefficients, label="Refit coefficients")
    if len(full) != len(refit):
        raise ValueError("Full and refit coefficient vectors must align")
    floor = float(denominator_floor)
    if not isfinite(floor) or floor <= 0.0:
        raise ValueError("Influence denominator floor must be positive")
    numerator = sqrt(
        sum((candidate - reference) ** 2 for reference, candidate in zip(
            full,
            refit,
            strict=True,
        ))
    )
    denominator = max(
        sqrt(sum(coefficient**2 for coefficient in full)),
        floor,
    )
    return numerator / denominator


def _sign_reversed(reference: float, candidate: float) -> bool:
    return (
        (reference > 0.0 and candidate < 0.0)
        or (reference < 0.0 and candidate > 0.0)
    )


def evaluate_influence_gate(
    full_coefficients: Sequence[float],
    full_p_values: Sequence[float],
    leave_one_refits: Sequence[Sequence[float]],
    leave_block_refits: Sequence[Sequence[float]],
    *,
    leave_one_threshold: float = OPEN02_LEAVE_ONE_INFLUENCE_THRESHOLD,
    leave_block_threshold: float = OPEN02_LEAVE_BLOCK_INFLUENCE_THRESHOLD,
    denominator_floor: float = OPEN02_INFLUENCE_DENOMINATOR_FLOOR,
    significance_alpha: float = OPEN02_SIGNIFICANCE_ALPHA,
) -> InfluenceGateResult:
    """Apply inclusive influence thresholds and significant-sign stability."""

    full = _finite_vector(full_coefficients, label="Full coefficients")
    p_values = _finite_vector(full_p_values, label="Full coefficient p-values")
    if len(full) != len(p_values):
        raise ValueError("Full coefficients and p-values must align")
    if any(value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("Full coefficient p-values must lie in [0, 1]")
    one_refits = [
        _finite_vector(row, label=f"Leave-one refit {index}")
        for index, row in enumerate(leave_one_refits)
    ]
    block_refits = [
        _finite_vector(row, label=f"Leave-block refit {index}")
        for index, row in enumerate(leave_block_refits)
    ]
    if not one_refits or not block_refits:
        raise ValueError("Influence evaluation requires both deletion families")
    if any(len(row) != len(full) for row in [*one_refits, *block_refits]):
        raise ValueError("Every influence refit must align with full coefficients")

    one_limit = float(leave_one_threshold)
    block_limit = float(leave_block_threshold)
    alpha = float(significance_alpha)
    if (
        not isfinite(one_limit)
        or not isfinite(block_limit)
        or one_limit < 0.0
        or block_limit < 0.0
    ):
        raise ValueError("Influence thresholds must be finite and nonnegative")
    if not isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("Significance alpha must lie in [0, 1]")

    one_influences = tuple(
        relative_l2_influence(
            full,
            row,
            denominator_floor=denominator_floor,
        )
        for row in one_refits
    )
    block_influences = tuple(
        relative_l2_influence(
            full,
            row,
            denominator_floor=denominator_floor,
        )
        for row in block_refits
    )
    maximum_one = max(one_influences)
    maximum_block = max(block_influences)
    one_passed = maximum_one <= one_limit
    block_passed = maximum_block <= block_limit
    significant_indices = {
        index for index, p_value in enumerate(p_values) if p_value <= alpha
    }
    sign_flip = any(
        _sign_reversed(full[index], refit[index])
        for refit in [*one_refits, *block_refits]
        for index in significant_indices
    )

    reasons: list[str] = []
    if not one_passed:
        reasons.append("leave_quarter_influence_gt_0_25")
    if not block_passed:
        reasons.append("leave_block_influence_gt_0_50")
    if sign_flip:
        reasons.append("sign_flip_under_influence")
    return InfluenceGateResult(
        leave_one_influences=one_influences,
        leave_block_influences=block_influences,
        maximum_leave_one_influence=maximum_one,
        maximum_leave_block_influence=maximum_block,
        leave_one_passed=one_passed,
        leave_block_passed=block_passed,
        sign_flip_detected=sign_flip,
        passed=not reasons,
        reason_codes=tuple(reasons),
    )


_QUARTER_PATTERN = re.compile(r"(?P<year>\d{4})Q(?P<quarter>[1-4])")


def _validation_failure(
    reason_code: str,
    message: str,
    **details: Any,
) -> Open02ValidationError:
    return Open02ValidationError(
        reason_code,
        message,
        details=details,
    )


def _quarter_ordinal(quarter: str) -> int:
    match = _QUARTER_PATTERN.fullmatch(str(quarter).strip())
    if match is None:
        raise _validation_failure(
            "coverage_gate_failed",
            f"Invalid quarter label: {quarter!r}",
            quarter=quarter,
        )
    return int(match.group("year")) * 4 + int(match.group("quarter")) - 1


def _quarter_from_ordinal(ordinal: int) -> str:
    year, zero_based_quarter = divmod(int(ordinal), 4)
    return f"{year:04d}Q{zero_based_quarter + 1}"


def _quarter_number(quarter: str) -> int:
    return _quarter_ordinal(quarter) % 4 + 1


def _quarter_from_row(row: Mapping[str, Any]) -> str:
    quarter = str(row.get("quarter", "")).strip()
    if quarter:
        _quarter_ordinal(quarter)
        return quarter
    period_end = str(row.get("period_end", "")).strip()
    try:
        parsed = datetime.strptime(period_end[:10], "%Y-%m-%d")
    except ValueError as exc:
        raise _validation_failure(
            "coverage_gate_failed",
            f"Row has no valid quarter or period_end: {period_end!r}",
            period_end=period_end,
        ) from exc
    expected_day = {3: 31, 6: 30, 9: 30, 12: 31}.get(parsed.month)
    if expected_day is None or parsed.day != expected_day:
        raise _validation_failure(
            "coverage_gate_failed",
            f"Period end is not a calendar quarter end: {period_end!r}",
            period_end=period_end,
        )
    return f"{parsed.year:04d}Q{parsed.month // 3}"


def _expected_quarters(contract: Open02Contract) -> tuple[str, ...]:
    start = _quarter_ordinal(contract.sample.start_quarter)
    end = _quarter_ordinal(contract.sample.end_quarter)
    quarters = tuple(_quarter_from_ordinal(value) for value in range(start, end + 1))
    if len(quarters) != contract.sample.observations:
        raise _validation_failure(
            "coverage_gate_failed",
            "Contract sample endpoints and observation count disagree",
            endpoints_observations=len(quarters),
            declared_observations=contract.sample.observations,
        )
    return quarters


def _quarter_hash(quarters: Sequence[str]) -> str:
    payload = "\n".join(str(quarter) for quarter in quarters)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_number(
    value: Any,
    *,
    reason_code: str,
    label: str,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _validation_failure(
            reason_code,
            f"{label} is not numeric",
            value=value,
        ) from exc
    if not isfinite(number):
        raise _validation_failure(
            reason_code,
            f"{label} is not finite",
            value=value,
        )
    return number


def _parse_utc_timestamp(value: Any, *, label: str) -> datetime:
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return datetime.strptime(text, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError as exc:
            raise _validation_failure(
                "vintage_gate_failed",
                f"{label} is not an ISO date",
                value=value,
            ) from exc
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _validation_failure(
            "vintage_gate_failed",
            f"{label} is not an ISO timestamp",
            value=value,
        ) from exc
    if parsed.tzinfo is None:
        raise _validation_failure(
            "vintage_gate_failed",
            f"{label} must carry a timezone",
            value=value,
        )
    return parsed.astimezone(timezone.utc)


def _normalize_source_bundle(
    source_bundle: Mapping[str, Any],
    *,
    contract: Open02Contract,
    quarters: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], str, dict[str, Any]]:
    if (
        source_bundle.get("schema_version") != 1
        or source_bundle.get("kind") != "open02_financial_accounts_input"
    ):
        raise _validation_failure(
            "metadata_gate_failed",
            "Source bundle schema/kind is not the frozen OPEN-02 input",
            schema_version=source_bundle.get("schema_version"),
            kind=source_bundle.get("kind"),
        )
    accepted_generated = str(
        source_bundle.get("accepted_tdcest_bundle_generated_at", "")
    ).strip()
    if accepted_generated != contract.sample.accepted_tdcest_bundle_generated_at:
        raise _validation_failure(
            "vintage_gate_failed",
            "Source bundle does not pin the accepted TDCest generation timestamp",
            expected=contract.sample.accepted_tdcest_bundle_generated_at,
            actual=accepted_generated,
        )

    bundle_vintage = str(source_bundle.get("observation_vintage", "")).strip()
    if bundle_vintage != contract.source.release_date:
        raise _validation_failure(
            "vintage_gate_failed",
            "Observation vintage must equal the pinned official release date",
            expected=contract.source.release_date,
            actual=bundle_vintage,
        )
    bundle_cutoff = str(
        source_bundle.get("observation_vintage_cutoff", "")
    ).strip()
    if bundle_cutoff != contract.sample.observation_vintage_cutoff:
        raise _validation_failure(
            "vintage_gate_failed",
            "Source bundle observation-vintage cutoff drifted",
            expected=contract.sample.observation_vintage_cutoff,
            actual=bundle_cutoff,
        )
    parsed_vintage = _parse_utc_timestamp(
        bundle_vintage,
        label="Bundle observation_vintage",
    )
    cutoff = _parse_utc_timestamp(
        contract.sample.observation_vintage_cutoff,
        label="Contract observation_vintage_cutoff",
    )
    if parsed_vintage > cutoff:
        raise _validation_failure(
            "vintage_gate_failed",
            "Observation vintage is later than the frozen cutoff",
            observation_vintage=bundle_vintage,
            cutoff=contract.sample.observation_vintage_cutoff,
        )

    official_release = source_bundle.get("official_release")
    if not isinstance(official_release, Mapping):
        raise _validation_failure(
            "metadata_gate_failed",
            "Source bundle has no official-release provenance",
        )
    expected_release_fields: dict[str, Any] = {
        "kind": "open02_board_z1_archive",
        "source_url": contract.source.archive_url,
        "release_date": contract.source.release_date,
        "observation_vintage_cutoff": (
            contract.sample.observation_vintage_cutoff
        ),
        "archive_sha256": contract.source.archive_sha256,
        "csv_member_sha256": dict(contract.source.csv_member_sha256),
        "dictionary_member_sha256": dict(
            contract.source.dictionary_member_sha256
        ),
        "sample_start": contract.sample.start_quarter,
        "sample_end": contract.sample.end_quarter,
        "observations": contract.sample.observations,
        "series_count": len(contract.series),
    }
    for field, expected_value in expected_release_fields.items():
        if official_release.get(field) != expected_value:
            raise _validation_failure(
                "metadata_gate_failed",
                f"Official-release provenance mismatch for {field}",
                field=field,
                expected=expected_value,
                actual=official_release.get(field),
            )
    declared_rows_hash = str(
        official_release.get("rows_sha256", "")
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", declared_rows_hash):
        raise _validation_failure(
            "metadata_gate_failed",
            "Official normalized rows hash is not a full SHA-256",
            actual=declared_rows_hash,
        )

    entries = source_bundle.get("series")
    if (
        not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes))
    ):
        raise _validation_failure(
            "metadata_gate_failed",
            "Source bundle series must be a sequence",
        )
    expected_by_key = {series.key: series for series in contract.series}
    actual_by_key: dict[str, Mapping[str, Any]] = {}
    duplicate_keys: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise _validation_failure(
                "metadata_gate_failed",
                "Every source series entry must be a mapping",
            )
        key = str(entry.get("key", "")).strip()
        if key in actual_by_key:
            duplicate_keys.append(key)
        actual_by_key[key] = entry
    if duplicate_keys:
        raise _validation_failure(
            "metadata_gate_failed",
            "Source bundle contains duplicate series keys",
            duplicate_keys=sorted(set(duplicate_keys)),
        )
    if set(actual_by_key) != set(expected_by_key):
        raise _validation_failure(
            "metadata_gate_failed",
            "Source bundle does not match the exact 20-series allowlist",
            missing=sorted(set(expected_by_key) - set(actual_by_key)),
            extra=sorted(set(actual_by_key) - set(expected_by_key)),
            expected_count=len(expected_by_key),
            actual_count=len(actual_by_key),
        )

    normalized: dict[str, dict[str, float]] = {}
    normalized_evidence: list[dict[str, Any]] = []
    expected_csv_members = dict(contract.source.csv_member_sha256)
    expected_dictionary_members = dict(
        contract.source.dictionary_member_sha256
    )
    source_metadata_fields = {
        "key",
        "fred_id",
        "board_series_id",
        "archive_member",
        "dictionary_member",
        "official_description",
        "table_line",
        "table",
        "unit_label",
        "side",
        "units",
        "seasonal_adjustment",
    }
    for key, expected in expected_by_key.items():
        entry = actual_by_key[key]
        expected_metadata = asdict(expected)
        for field, expected_value in expected_metadata.items():
            actual_value = entry.get(field)
            if field == "treasury_lineage" and isinstance(
                actual_value, Sequence
            ) and not isinstance(actual_value, (str, bytes)):
                actual_value = tuple(actual_value)
            if actual_value != expected_value:
                raise _validation_failure(
                    "metadata_gate_failed",
                    f"Metadata mismatch for {key}.{field}",
                    series_key=key,
                    field=field,
                    expected=expected_value,
                    actual=actual_value,
                )
        source_metadata = entry.get("source_metadata")
        if not isinstance(source_metadata, Mapping):
            raise _validation_failure(
                "metadata_gate_failed",
                f"{key} has no official source metadata",
                series_key=key,
            )
        if set(source_metadata) != source_metadata_fields:
            raise _validation_failure(
                "metadata_gate_failed",
                f"{key} official source metadata fields drifted",
                series_key=key,
                missing=sorted(source_metadata_fields - set(source_metadata)),
                extra=sorted(set(source_metadata) - source_metadata_fields),
            )
        expected_description = expected.official_title.removesuffix(
            ", Transactions"
        )
        expected_source_metadata = {
            "key": expected.key,
            "fred_id": expected.fred_id,
            "board_series_id": expected.board_series_id,
            "official_description": expected_description,
            "unit_label": contract.source.unit_label,
            "side": expected.side,
            "units": expected.units,
            "seasonal_adjustment": expected.seasonal_adjustment,
        }
        for field, expected_value in expected_source_metadata.items():
            actual_value = source_metadata.get(field)
            equal = (
                str(actual_value).casefold() == str(expected_value).casefold()
                if field == "official_description"
                else actual_value == expected_value
            )
            if not equal:
                raise _validation_failure(
                    "metadata_gate_failed",
                    f"Official source metadata mismatch for {key}.{field}",
                    series_key=key,
                    field=field,
                    expected=expected_value,
                    actual=actual_value,
                )
        archive_member = str(source_metadata.get("archive_member", ""))
        dictionary_member = str(source_metadata.get("dictionary_member", ""))
        if (
            archive_member not in expected_csv_members
            or dictionary_member not in expected_dictionary_members
            or dictionary_member
            != archive_member.replace("csv/", "data_dictionary/").removesuffix(
                ".csv"
            )
            + ".txt"
        ):
            raise _validation_failure(
                "metadata_gate_failed",
                f"{key} archive/dictionary membership is not pinned",
                series_key=key,
                archive_member=archive_member,
                dictionary_member=dictionary_member,
            )
        if (
            not str(source_metadata.get("table_line", "")).strip()
            or not str(source_metadata.get("table", "")).strip()
        ):
            raise _validation_failure(
                "metadata_gate_failed",
                f"{key} lacks Board table/line metadata",
                series_key=key,
            )
        entry_vintage = str(entry.get("observation_vintage", "")).strip()
        if entry_vintage != bundle_vintage:
            raise _validation_failure(
                "vintage_gate_failed",
                "All 20 series must carry one identical observation vintage",
                series_key=key,
                bundle_vintage=bundle_vintage,
                series_vintage=entry_vintage,
            )

        observations = entry.get("observations")
        if (
            not isinstance(observations, Sequence)
            or isinstance(observations, (str, bytes))
        ):
            raise _validation_failure(
                "coverage_gate_failed",
                f"{key} observations must be a sequence",
                series_key=key,
            )
        values: dict[str, float] = {}
        duplicates: list[str] = []
        for row in observations:
            if not isinstance(row, Mapping):
                raise _validation_failure(
                    "coverage_gate_failed",
                    f"{key} contains a non-mapping observation",
                    series_key=key,
                )
            quarter = _quarter_from_row(row)
            if quarter in values:
                duplicates.append(quarter)
            values[quarter] = _finite_number(
                row.get("value"),
                reason_code="coverage_gate_failed",
                label=f"{key} value for {quarter}",
            )
        if duplicates or tuple(sorted(values, key=_quarter_ordinal)) != quarters:
            raise _validation_failure(
                "coverage_gate_failed",
                f"{key} does not contain the exact ordered 96-quarter sample",
                series_key=key,
                duplicates=sorted(set(duplicates), key=_quarter_ordinal),
                missing=sorted(set(quarters) - set(values), key=_quarter_ordinal),
                extra=sorted(set(values) - set(quarters), key=_quarter_ordinal),
                actual_count=len(observations),
            )
        normalized[key] = values
        normalized_evidence.append(
            {
                "key": key,
                "metadata": expected_metadata,
                "source_metadata": dict(source_metadata),
                "observation_vintage": entry_vintage,
                "observations": [
                    {"quarter": quarter, "value": values[quarter]}
                    for quarter in quarters
                ],
            }
        )

    normalized_wide_rows = [
        {
            "quarter": quarter,
            **{
                series.key: normalized[series.key][quarter]
                for series in contract.series
            },
        }
        for quarter in quarters
    ]
    actual_rows_hash = _stable_hash(normalized_wide_rows)
    if actual_rows_hash != declared_rows_hash:
        raise _validation_failure(
            "coverage_gate_failed",
            "Official normalized rows do not match their declared SHA-256",
            expected=declared_rows_hash,
            actual=actual_rows_hash,
        )
    evidence = {
        "series_count": len(normalized),
        "series_keys": list(expected_by_key),
        "observation_vintage": bundle_vintage,
        "official_release": dict(official_release),
        "normalized_rows_sha256": actual_rows_hash,
        "source_bundle_sha256": _stable_hash(
            {
                "schema_version": 1,
                "kind": "open02_financial_accounts_input",
                "accepted_tdcest_bundle_generated_at": accepted_generated,
                "observation_vintage": bundle_vintage,
                "observation_vintage_cutoff": bundle_cutoff,
                "official_release": dict(official_release),
                "series": normalized_evidence,
            }
        ),
    }
    return normalized, bundle_vintage, evidence


def _normalize_standardized_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Open02Contract,
    quarters: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], str]:
    required_ids = (
        contract.canonical_treatment_source_series,
        contract.embedded_bank_treasury_component_id,
    )
    target_set = set(quarters)
    start_ordinal = _quarter_ordinal(contract.sample.start_quarter)
    end_ordinal = _quarter_ordinal(contract.sample.end_quarter)
    normalized = {series_id: {} for series_id in required_ids}
    duplicates: dict[str, list[str]] = {series_id: [] for series_id in required_ids}
    evidence_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise _validation_failure(
                "coverage_gate_failed",
                "Every accepted standardized row must be a mapping",
            )
        series_id = str(row.get("series_id", "")).strip()
        if series_id not in normalized:
            continue
        quarter = _quarter_from_row(row)
        if _quarter_ordinal(quarter) > end_ordinal:
            raise _validation_failure(
                "coverage_gate_failed",
                "Accepted standardized inputs contain a post-sample row",
                series_id=series_id,
                quarter=quarter,
            )
        if _quarter_ordinal(quarter) < start_ordinal:
            continue
        if str(row.get("freq", "quarterly")).strip() != "quarterly":
            raise _validation_failure(
                "metadata_gate_failed",
                "Accepted standardized inputs must be quarterly",
                series_id=series_id,
                quarter=quarter,
            )
        if str(row.get("units", "")).strip() != "usd_millions":
            raise _validation_failure(
                "metadata_gate_failed",
                "Accepted treatment/component rows must use usd_millions",
                series_id=series_id,
                quarter=quarter,
                units=row.get("units"),
            )
        if quarter in normalized[series_id]:
            duplicates[series_id].append(quarter)
        value = _finite_number(
            row.get("value"),
            reason_code="coverage_gate_failed",
            label=f"{series_id} accepted value for {quarter}",
        )
        normalized[series_id][quarter] = value
        if quarter in target_set:
            evidence_rows.append(
                {"series_id": series_id, "quarter": quarter, "value": value}
            )
    for series_id in required_ids:
        actual = normalized[series_id]
        target_actual = {quarter: actual[quarter] for quarter in actual if quarter in target_set}
        if duplicates[series_id] or set(target_actual) != target_set:
            raise _validation_failure(
                "coverage_gate_failed",
                "Accepted standardized series do not cover the exact target sample",
                series_id=series_id,
                duplicates=sorted(
                    set(duplicates[series_id]),
                    key=_quarter_ordinal,
                ),
                missing=sorted(target_set - set(target_actual), key=_quarter_ordinal),
            )
        normalized[series_id] = target_actual
    evidence_rows.sort(key=lambda row: (row["series_id"], _quarter_ordinal(row["quarter"])))
    return normalized, _stable_hash(evidence_rows)


def _normalize_design_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Open02Contract,
    quarters: tuple[str, ...],
) -> tuple[tuple[dict[str, Any], ...], str]:
    control_specs = contract.control_specs
    if tuple(spec.control_id for spec in control_specs) != contract.control_ids:
        raise _validation_failure(
            "common_sample_design_failed",
            "Typed control registry and frozen control IDs disagree",
        )
    if any(spec.treasury_lineage != (0, 0, 0) for spec in control_specs):
        raise _validation_failure(
            "lineage_gate_failed",
            "Every OPEN-02 control must have zero Treasury lineage",
        )
    direct_controls: dict[str, str] = {}
    derived_controls: dict[str, tuple[str, int]] = {}
    quarter_dummies: dict[str, int] = {}
    for spec in control_specs:
        if spec.indicator_quarter is not None:
            if (
                spec.source_series_id is not None
                or spec.lag_quarters != 0
                or spec.indicator_quarter not in (2, 3, 4)
            ):
                raise _validation_failure(
                    "common_sample_design_failed",
                    "Quarter-indicator control metadata is inconsistent",
                    control_id=spec.control_id,
                )
            quarter_dummies[spec.control_id] = spec.indicator_quarter
        elif spec.lag_quarters > 0:
            if not spec.source_series_id:
                raise _validation_failure(
                    "common_sample_design_failed",
                    "Lagged control has no typed source series",
                    control_id=spec.control_id,
                )
            derived_controls[spec.control_id] = (
                spec.source_series_id,
                spec.lag_quarters,
            )
        else:
            if not spec.source_series_id or spec.lag_quarters != 0:
                raise _validation_failure(
                    "common_sample_design_failed",
                    "Direct control metadata is inconsistent",
                    control_id=spec.control_id,
                )
            direct_controls[spec.control_id] = spec.source_series_id

    history: dict[str, Mapping[str, Any]] = {}
    end_ordinal = _quarter_ordinal(contract.sample.end_quarter)
    for row in rows:
        if not isinstance(row, Mapping):
            raise _validation_failure(
                "coverage_gate_failed",
                "Every accepted design row must be a mapping",
            )
        quarter = _quarter_from_row(row)
        if _quarter_ordinal(quarter) > end_ordinal:
            continue
        if quarter in history:
            raise _validation_failure(
                "coverage_gate_failed",
                "Accepted design history contains a duplicate quarter",
                quarter=quarter,
            )
        history[quarter] = row

    target_rows: list[dict[str, Any]] = []
    for quarter in quarters:
        source = history.get(quarter)
        if source is None:
            raise _validation_failure(
                "coverage_gate_failed",
                "Accepted design history misses a target quarter",
                quarter=quarter,
            )
        output: dict[str, Any] = {"quarter": quarter}
        canonical_design_id = contract.open01_contract.canonical_treatment_id
        output[canonical_design_id] = _finite_number(
            source.get(canonical_design_id),
            reason_code="coverage_gate_failed",
            label=f"{canonical_design_id} design value for {quarter}",
        )
        for control_id, source_id in direct_controls.items():
            output[control_id] = _finite_number(
                source.get(source_id),
                reason_code="coverage_gate_failed",
                label=f"{source_id} design value for {quarter}",
            )
        ordinal = _quarter_ordinal(quarter)
        for control_id, (source_id, lag) in derived_controls.items():
            lag_quarter = _quarter_from_ordinal(ordinal - lag)
            lag_row = history.get(lag_quarter)
            if lag_row is None:
                raise _validation_failure(
                    "coverage_gate_failed",
                    f"Full accepted history misses {source_id} lag source",
                    control_id=control_id,
                    quarter=quarter,
                    required_quarter=lag_quarter,
                )
            output[control_id] = _finite_number(
                lag_row.get(source_id),
                reason_code="coverage_gate_failed",
                label=f"{source_id} value for {lag_quarter}",
            )
        quarter_number = _quarter_number(quarter)
        for control_id, dummy_quarter in quarter_dummies.items():
            output[control_id] = float(quarter_number == dummy_quarter)
        target_rows.append(output)
    return tuple(target_rows), _stable_hash(target_rows)


def _max_abs_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return max(
        abs(a - b)
        for a, b in zip(left, right, strict=True)
    )


def _assert_tolerance(
    *,
    gate_reason: str,
    label: str,
    residual: float,
    tolerance: float,
) -> None:
    if residual > tolerance:
        raise _validation_failure(
            gate_reason,
            f"{label} exceeds the frozen tolerance",
            maximum_absolute_residual=residual,
            tolerance=tolerance,
        )


def _validate_lineage(contract: Open02Contract) -> dict[str, Any]:
    lineage_items = contract.lineage.lineage_by_id
    lineage_by_id = dict(lineage_items)
    if len(lineage_by_id) != len(lineage_items):
        raise _validation_failure(
            "lineage_gate_failed",
            "Lineage registry contains duplicate identifiers",
        )
    series_by_key = {series.key: series for series in contract.series}
    nonzero_fred_ids = tuple(
        series.fred_id
        for series in contract.series
        if any(series.treasury_lineage)
    )
    if nonzero_fred_ids != contract.lineage.raw_treasury_ids:
        raise _validation_failure(
            "lineage_gate_failed",
            "Raw Treasury lineage identifiers do not match the frozen allowlist",
            expected=list(contract.lineage.raw_treasury_ids),
            actual=list(nonzero_fred_ids),
        )
    for control in contract.control_specs:
        if lineage_by_id.get(control.control_id) != control.treasury_lineage:
            raise _validation_failure(
                "lineage_gate_failed",
                "Typed control lineage and equation lineage registry disagree",
                control_id=control.control_id,
                control_lineage=control.treasury_lineage,
                registry_lineage=lineage_by_id.get(control.control_id),
            )

    canonical_ids = {
        contract.open01_contract.canonical_treatment_id,
        contract.canonical_treatment_source_series,
    }
    checked_equations = 0
    for system in contract.systems:
        if canonical_ids.intersection(system.regressor_ids):
            raise _validation_failure(
                "lineage_gate_failed",
                "Canonical TDC is prohibited on an OPEN-02 right-hand side",
                system_id=system.system_id,
            )
        outcome_ids = (*system.outcome_ids, *system.agency_component_outcome_ids)
        rhs_ids = (*system.regressor_ids, *contract.control_ids)
        for outcome_id in outcome_ids:
            lhs = lineage_by_id.get(
                outcome_id,
                series_by_key.get(outcome_id).treasury_lineage
                if outcome_id in series_by_key
                else None,
            )
            if lhs is None:
                raise _validation_failure(
                    "lineage_gate_failed",
                    "Outcome has no declared Treasury lineage",
                    system_id=system.system_id,
                    outcome_id=outcome_id,
                )
            for regressor_id in rhs_ids:
                rhs = lineage_by_id.get(regressor_id)
                if rhs is None:
                    raise _validation_failure(
                        "lineage_gate_failed",
                        "Right-hand-side variable has no declared Treasury lineage",
                        system_id=system.system_id,
                        regressor_id=regressor_id,
                    )
                if any(left and right for left, right in zip(lhs, rhs, strict=True)):
                    raise _validation_failure(
                        "lineage_gate_failed",
                        "A raw Treasury lineage appears on both sides of an equation",
                        system_id=system.system_id,
                        outcome_id=outcome_id,
                        regressor_id=regressor_id,
                        outcome_lineage=lhs,
                        regressor_lineage=rhs,
                    )
            checked_equations += 1
    return {
        "checked_equations": checked_equations,
        "raw_treasury_ids": list(contract.lineage.raw_treasury_ids),
        "lineage_by_id": {
            identifier: list(lineage)
            for identifier, lineage in lineage_items
        },
    }


def _normal_p_value(coefficient: float, standard_error: float) -> float:
    if not isfinite(standard_error) or standard_error <= 0.0:
        raise _validation_failure(
            "rank_gate_failed",
            "Inference requires a finite positive standard error",
            coefficient=coefficient,
            standard_error=standard_error,
        )
    return erfc(abs(coefficient / standard_error) / sqrt(2.0))


def _estimate_row(
    *,
    system: StackedSystemFit,
    system_id: str,
    outcome_id: str,
    equation_index: int,
    coefficient_index: int,
    coefficient_id: str,
    regressor_id: str,
    sample_hash: str,
    design_hash: str,
    descriptive_only: bool,
    covariance_contract: Open02CovarianceContract,
) -> dict[str, Any]:
    flat_index = system.flat_index(equation_index, coefficient_index)
    estimate = system.equation_fits[equation_index].coefficients[coefficient_index]
    variance = system.covariance[flat_index][flat_index]
    if not isfinite(variance) or variance <= 0.0:
        raise _validation_failure(
            "rank_gate_failed",
            "Stacked HAC produced a nonpositive coefficient variance",
            system_id=system_id,
            outcome_id=outcome_id,
            coefficient_id=coefficient_id,
            variance=variance,
        )
    standard_error = sqrt(variance)
    return {
        "system_id": system_id,
        "outcome_id": outcome_id,
        "coefficient_id": coefficient_id,
        "regressor_id": regressor_id,
        "estimate": estimate,
        "standard_error": standard_error,
        "raw_p_value": _normal_p_value(estimate, standard_error),
        "descriptive_only": descriptive_only,
        "observations": system.observations,
        "parameters": system.parameters_per_equation,
        "sample_hash": sample_hash,
        "design_hash": design_hash,
        "coefficient_estimator": covariance_contract.coefficient_estimator,
        "covariance_estimator": covariance_contract.estimator,
        "kernel": system.kernel,
        "hac_lags": system.hac_lags,
        "prewhitened": system.prewhitened,
        "prewhitening": covariance_contract.prewhitening,
        "finite_sample_correction": (
            covariance_contract.finite_sample_correction
        ),
        "finite_sample_scale": system.finite_sample_scale,
        "test_sidedness": covariance_contract.test_sidedness,
        "score_definition": covariance_contract.score_definition,
        "sandwich_bread": covariance_contract.sandwich_bread,
    }


def _relative_adding_up_error(aggregate: float, components: Sequence[float]) -> float:
    component_sum = sum(components)
    return abs(aggregate - component_sum) / max(
        1.0,
        abs(aggregate),
        abs(component_sum),
    )


def _refit_equations(
    outcomes_by_equation: Sequence[Sequence[float]],
    x_rows: Sequence[Sequence[float]],
) -> tuple[OLSFit, ...]:
    design = _validated_design(x_rows)
    try:
        inverse = _design_inverse(design)
    except ValueError as exc:
        raise _validation_failure(
            "rank_gate_failed",
            "An OPEN-02 deletion design is not full rank",
        ) from exc
    outcomes = [
        _finite_vector(values, label=f"Refit equation {index}")
        for index, values in enumerate(outcomes_by_equation)
    ]
    if any(len(values) != len(design) for values in outcomes):
        raise ValueError("Refit outcomes and design do not align")
    return tuple(
        _fit_with_inverse(values, design, inverse)
        for values in outcomes
    )


def evaluate_open02_eligibility(
    wald_rows: Sequence[Mapping[str, Any]],
    influence_summaries: Sequence[Mapping[str, Any]],
    *,
    contract: Open02Contract = OPEN02_CONTRACT,
) -> dict[str, Any]:
    """Apply the frozen statistical promotion rule to a valid result."""

    wald_by_id: dict[str, Mapping[str, Any]] = {}
    for row in wald_rows:
        hypothesis_id = str(row.get("hypothesis_id", ""))
        if hypothesis_id in wald_by_id:
            raise ValueError(f"Duplicate Wald row: {hypothesis_id}")
        wald_by_id[hypothesis_id] = row
    if set(wald_by_id) != set(contract.holm.hypothesis_ids):
        raise ValueError("Eligibility requires exactly H_T, H_P, and H_W")
    influence_by_group: dict[str, Mapping[str, Any]] = {}
    for row in influence_summaries:
        group_id = str(row.get("group_id", ""))
        if group_id in influence_by_group:
            raise ValueError(f"Duplicate influence summary: {group_id}")
        influence_by_group[group_id] = row
    expected_group_ids = {
        group.group_id for group in contract.influence_groups
    }
    if set(influence_by_group) != expected_group_ids:
        raise ValueError(
            "Eligibility requires exactly the contract influence groups"
        )

    failed: set[str] = set()
    hypothesis_reason = {
        "H_T": "source_anchor_holm_gt_0_05",
        "H_P": "portfolio_joint_holm_gt_0_05",
        "H_W": "within_joint_holm_gt_0_05",
    }
    for hypothesis_id in contract.holm.hypothesis_ids:
        adjusted = _finite_number(
            wald_by_id[hypothesis_id].get("holm_adjusted_p_value"),
            reason_code="statistical_gate_invalid",
            label=f"{hypothesis_id} Holm-adjusted p-value",
        )
        if not 0.0 <= adjusted <= 1.0:
            raise ValueError(
                f"{hypothesis_id} Holm-adjusted p-value must lie in [0, 1]"
            )
        if adjusted > contract.holm.familywise_alpha:
            failed.add(hypothesis_reason[hypothesis_id])
    influence_reason_codes = set(contract.promotion_reason_codes) - set(
        hypothesis_reason.values()
    )
    for row in influence_by_group.values():
        row_reasons = tuple(str(reason) for reason in row.get("reason_codes", ()))
        if (
            not set(row_reasons).issubset(influence_reason_codes)
            or bool(row.get("passed")) != (not row_reasons)
        ):
            raise ValueError(
                "Influence summary pass flag and reason codes are inconsistent"
            )
        failed.update(row_reasons)

    ordered_reasons = tuple(
        reason
        for reason in contract.promotion_reason_codes
        if reason in failed
    )
    eligible = not ordered_reasons
    disposition = (
        contract.promoted_result_disposition
        if eligible
        else contract.valid_nonpromoted_disposition
    )
    return {
        "valid_result": disposition.valid_result,
        "main_text_eligible": disposition.main_text_eligible,
        "appendix_only": disposition.appendix_only,
        "reason_codes": ordered_reasons,
        "holm_familywise_alpha": contract.holm.familywise_alpha,
    }


def run_open02_pipeline(
    source_bundle: Mapping[str, Any],
    accepted_design_rows: Sequence[Mapping[str, Any]],
    accepted_standardized_rows: Sequence[Mapping[str, Any]],
    *,
    contract: Open02Contract = OPEN02_CONTRACT,
) -> Open02PipelineResult:
    """Validate, estimate, and evaluate one complete frozen OPEN-02 bundle."""

    if (
        len(contract.validity_gates) != 12
        or len({gate.gate_id for gate in contract.validity_gates}) != 12
        or len({gate.reason_code for gate in contract.validity_gates}) != 12
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "OPEN-02 must declare exactly 12 unique deterministic validity gates",
        )
    supported_covariance = {
        "coefficient_estimator": "equation_by_equation_ols",
        "estimator": "stacked_system_newey_west_hac",
        "kernel": "bartlett",
        "prewhitening": "none",
        "finite_sample_correction": "T/(T-K)",
        "test_sidedness": "two_sided",
        "score_definition": (
            "equation_major_stack_of_x_t_times_equation_residual"
        ),
        "sandwich_bread": (
            "block_diagonal_copies_of_inverse_X_transpose_X"
        ),
    }
    covariance_values = asdict(contract.covariance)
    if any(
        covariance_values[field] != expected
        for field, expected in supported_covariance.items()
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "The covariance contract is not implemented by the live estimator",
            actual=covariance_values,
        )
    if (
        len(contract.holm.hypothesis_ids) != OPEN02_HOLM_FAMILY_SIZE
        or len(contract.wald_hypotheses) != OPEN02_HOLM_FAMILY_SIZE
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "OPEN-02 requires exactly three declared Wald/Holm hypotheses",
        )

    quarters = _expected_quarters(contract)
    sample_hash = _quarter_hash(quarters)
    if sample_hash != contract.sample.quarter_hash:
        raise _validation_failure(
            "coverage_gate_failed",
            "Ordered sample hash differs from the frozen OPEN-02 contract",
            expected=contract.sample.quarter_hash,
            actual=sample_hash,
        )
    source, observation_vintage, source_evidence = _normalize_source_bundle(
        source_bundle,
        contract=contract,
        quarters=quarters,
    )
    standardized, standardized_hash = _normalize_standardized_rows(
        accepted_standardized_rows,
        contract=contract,
        quarters=quarters,
    )
    design_rows, accepted_design_hash = _normalize_design_rows(
        accepted_design_rows,
        contract=contract,
        quarters=quarters,
    )
    source_system_candidates = [
        system
        for system in contract.systems
        if system.regressor_ids == (OPEN02_LEAVE_OUT_ID,)
        and OPEN02_BANK_TREASURY_ID in system.outcome_ids
    ]
    within_system_candidates = [
        system
        for system in contract.systems
        if system.regressor_ids
        == (OPEN02_BANK_TREASURY_ID, OPEN02_LEAVE_OUT_ID)
    ]
    if (
        len(source_system_candidates) != 1
        or len(within_system_candidates) != 1
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "OPEN-02 source and within systems are not uniquely declared",
        )
    source_system_contract = source_system_candidates[0]
    within_system_contract = within_system_candidates[0]
    agency_component_ids = source_system_contract.agency_component_outcome_ids
    if (
        len(agency_component_ids) != 7
        or within_system_contract.agency_component_outcome_ids
        != agency_component_ids
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "Source and within systems must share the same seven agency components",
        )
    formula_by_id = {formula.formula_id: formula for formula in contract.formulas}
    if len(formula_by_id) != len(contract.formulas):
        raise _validation_failure(
            "common_sample_design_failed",
            "OPEN-02 formula registry contains duplicate IDs",
        )
    required_formula_ids = (
        "bank_treasury_three_sector_sum",
        "accepted_bank_treasury_component_reconciliation",
        "leave_out_definition",
        "canonical_reconstruction",
        "agency_us_five_component_identity",
        "agency_three_sector_identity",
        "agency_seven_component_identity",
        "loans_three_sector_identity",
        "customer_deposit_identity",
        "source_agency_coefficient_adding_up",
        "within_agency_coefficient_adding_up",
    )
    if not set(required_formula_ids).issubset(formula_by_id):
        raise _validation_failure(
            "common_sample_design_failed",
            "OPEN-02 formula registry is incomplete",
            missing=sorted(set(required_formula_ids) - set(formula_by_id)),
        )

    canonical_id = contract.canonical_treatment_source_series
    canonical_design_id = contract.open01_contract.canonical_treatment_id
    accepted_bank_id = contract.embedded_bank_treasury_component_id
    identity_formula_ids = required_formula_ids[:9]
    if any(
        formula_by_id[formula_id].tolerance_kind
        != "absolute_usd_millions"
        or formula_by_id[formula_id].tolerance
        != contract.sample.identity_tolerance_usd_millions
        for formula_id in identity_formula_ids
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "Identity formulas and the sample tolerance do not align",
        )
    identity_tolerances = {
        "treasury_component": formula_by_id[
            "bank_treasury_three_sector_sum"
        ].tolerance,
        "accepted_component_reconciliation": formula_by_id[
            "accepted_bank_treasury_component_reconciliation"
        ].tolerance,
        "leave_out_reconstruction": formula_by_id[
            "canonical_reconstruction"
        ].tolerance,
        "us_agency_identity": formula_by_id[
            "agency_us_five_component_identity"
        ].tolerance,
        "three_sector_agency_identity": formula_by_id[
            "agency_seven_component_identity"
        ].tolerance,
    }
    canonical_standardized = [
        standardized[canonical_id][quarter] for quarter in quarters
    ]
    canonical_design = [float(row[canonical_design_id]) for row in design_rows]
    canonical_cross_surface_residual = _max_abs_difference(
        canonical_standardized,
        canonical_design,
    )
    _assert_tolerance(
        gate_reason="leave_out_reconstruction_failed",
        label="Canonical accepted design/standardized reconciliation",
        residual=canonical_cross_surface_residual,
        tolerance=identity_tolerances["leave_out_reconstruction"],
    )

    treasury_formula = formula_by_id["bank_treasury_three_sector_sum"]
    us_agency_formula = formula_by_id[
        "agency_us_five_component_identity"
    ]
    agency_formula = formula_by_id["agency_three_sector_identity"]
    agency_component_formula = formula_by_id[
        "agency_seven_component_identity"
    ]
    loan_formula = formula_by_id["loans_three_sector_identity"]
    deposit_formula = formula_by_id["customer_deposit_identity"]
    panel_rows: list[dict[str, Any]] = []
    accepted_component_residuals: list[float] = []
    reconstruction_residuals: list[float] = []
    us_agency_residuals: list[float] = []
    three_sector_agency_residuals: list[float] = []
    treasury_definition_residuals: list[float] = []
    for index, quarter in enumerate(quarters):
        raw = {key: source[key][quarter] for key in source}
        bank_treasury = sum(
            coefficient * raw[series_id]
            for series_id, coefficient in treasury_formula.terms
        )
        accepted_bank_treasury = standardized[accepted_bank_id][quarter]
        canonical = canonical_standardized[index]
        leave_out = canonical - bank_treasury
        us_agency_sum = sum(
            coefficient * raw[series_id]
            for series_id, coefficient in us_agency_formula.terms
        )
        agency = sum(
            coefficient * raw[series_id]
            for series_id, coefficient in agency_formula.terms
        )
        agency_components_sum = sum(
            coefficient * raw[series_id]
            for series_id, coefficient in agency_component_formula.terms
        )
        loans = sum(
            coefficient * raw[series_id]
            for series_id, coefficient in loan_formula.terms
        )
        deposits = sum(
            coefficient * raw[series_id]
            for series_id, coefficient in deposit_formula.terms
        )
        treasury_definition_residuals.append(
            bank_treasury
            - sum(
                coefficient * raw[series_id]
                for series_id, coefficient in treasury_formula.terms
            )
        )
        accepted_component_residuals.append(
            bank_treasury - accepted_bank_treasury
        )
        reconstruction_residuals.append(canonical - leave_out - bank_treasury)
        us_agency_residuals.append(raw["agency_us_total"] - us_agency_sum)
        three_sector_agency_residuals.append(agency - agency_components_sum)
        row: dict[str, Any] = {
            "quarter": quarter,
            "C": canonical,
            "X": leave_out,
            "B": bank_treasury,
            "A": agency,
            "L": loans,
            "D": deposits,
            canonical_id: canonical,
            canonical_design_id: canonical,
            accepted_bank_id: accepted_bank_treasury,
            OPEN02_LEAVE_OUT_ID: leave_out,
            OPEN02_BANK_TREASURY_ID: bank_treasury,
            OPEN02_BANK_AGENCY_ID: agency,
            OPEN02_BANK_LOANS_ID: loans,
            OPEN02_BANK_DEPOSITS_ID: deposits,
            **raw,
        }
        for control_id in contract.control_ids:
            if control_id not in design_rows[index]:
                raise _validation_failure(
                    "common_sample_design_failed",
                    "A frozen OPEN-02 control is absent from the derived design",
                    control_id=control_id,
                    quarter=quarter,
                )
            row[control_id] = float(design_rows[index][control_id])
        panel_rows.append(row)

    identity_residuals = {
        "treasury_component": max(abs(value) for value in treasury_definition_residuals),
        "accepted_component_reconciliation": max(
            abs(value) for value in accepted_component_residuals
        ),
        "leave_out_reconstruction": max(
            canonical_cross_surface_residual,
            max(abs(value) for value in reconstruction_residuals),
        ),
        "us_agency_identity": max(abs(value) for value in us_agency_residuals),
        "three_sector_agency_identity": max(
            abs(value) for value in three_sector_agency_residuals
        ),
    }
    for gate_id, reason_code in (
        ("treasury_component", "treasury_component_gate_failed"),
        (
            "accepted_component_reconciliation",
            "accepted_component_reconciliation_failed",
        ),
        ("leave_out_reconstruction", "leave_out_reconstruction_failed"),
        ("us_agency_identity", "us_agency_identity_failed"),
        (
            "three_sector_agency_identity",
            "three_sector_agency_identity_failed",
        ),
    ):
        _assert_tolerance(
            gate_reason=reason_code,
            label=gate_id,
            residual=identity_residuals[gate_id],
            tolerance=identity_tolerances[gate_id],
        )

    lineage_evidence = _validate_lineage(contract)
    source_columns = (
        "intercept",
        OPEN02_LEAVE_OUT_ID,
        *contract.control_ids,
    )
    within_columns = (
        "intercept",
        OPEN02_BANK_TREASURY_ID,
        OPEN02_LEAVE_OUT_ID,
        *contract.control_ids,
    )
    if len(source_columns) != source_system_contract.design_parameter_count:
        raise _validation_failure(
            "common_sample_design_failed",
            "Source-system design column count differs from the contract",
        )
    if len(within_columns) != within_system_contract.design_parameter_count:
        raise _validation_failure(
            "common_sample_design_failed",
            "Within-system design column count differs from the contract",
        )
    source_design = [
        [
            1.0,
            row[OPEN02_LEAVE_OUT_ID],
            *(row[control_id] for control_id in contract.control_ids),
        ]
        for row in panel_rows
    ]
    within_design = [
        [
            1.0,
            row[OPEN02_BANK_TREASURY_ID],
            row[OPEN02_LEAVE_OUT_ID],
            *(row[control_id] for control_id in contract.control_ids),
        ]
        for row in panel_rows
    ]
    try:
        _design_inverse(_validated_design(source_design))
        _design_inverse(_validated_design(within_design))
    except ValueError as exc:
        raise _validation_failure(
            "rank_gate_failed",
            "A frozen OPEN-02 full-sample design is not full rank",
        ) from exc
    source_design_hash = _stable_hash(
        {"columns": source_columns, "rows": source_design}
    )
    within_design_hash = _stable_hash(
        {"columns": within_columns, "rows": within_design}
    )
    row_hash = _quarter_hash([row["quarter"] for row in panel_rows])
    if row_hash != sample_hash:
        raise _validation_failure(
            "common_sample_design_failed",
            "Panel row hash changed during construction",
        )

    source_aggregate_outcomes = [
        [row[outcome_id] for row in panel_rows]
        for outcome_id in source_system_contract.outcome_ids
    ]
    source_component_outcomes = [
        [row[outcome_id] for row in panel_rows]
        for outcome_id in source_system_contract.agency_component_outcome_ids
    ]
    within_aggregate_outcomes = [
        [row[outcome_id] for row in panel_rows]
        for outcome_id in within_system_contract.outcome_ids
    ]
    within_component_outcomes = [
        [row[outcome_id] for row in panel_rows]
        for outcome_id in within_system_contract.agency_component_outcome_ids
    ]
    try:
        source_aggregate_fit = fit_stacked_system(
            source_aggregate_outcomes,
            source_design,
            hac_lags=contract.covariance.lag_quarters,
        )
        source_component_fit = fit_stacked_system(
            source_component_outcomes,
            source_design,
            hac_lags=contract.covariance.lag_quarters,
        )
        within_aggregate_fit = fit_stacked_system(
            within_aggregate_outcomes,
            within_design,
            hac_lags=contract.covariance.lag_quarters,
        )
        within_component_fit = fit_stacked_system(
            within_component_outcomes,
            within_design,
            hac_lags=contract.covariance.lag_quarters,
        )
    except ValueError as exc:
        raise _validation_failure(
            "rank_gate_failed",
            "OPEN-02 system estimation failed",
            error=str(exc),
        ) from exc

    estimate_rows: list[dict[str, Any]] = []
    for equation_index, (outcome_id, coefficient_id) in enumerate(
        zip(
            source_system_contract.outcome_ids,
            source_system_contract.coefficient_ids,
            strict=True,
        )
    ):
        estimate_rows.append(
            _estimate_row(
                system=source_aggregate_fit,
                system_id=source_system_contract.system_id,
                outcome_id=outcome_id,
                equation_index=equation_index,
                coefficient_index=1,
                coefficient_id=coefficient_id,
                regressor_id=source_system_contract.regressor_ids[0],
                sample_hash=sample_hash,
                design_hash=source_design_hash,
                descriptive_only=False,
                covariance_contract=contract.covariance,
            )
        )
    for equation_index, outcome_id in enumerate(agency_component_ids):
        estimate_rows.append(
            _estimate_row(
                system=source_component_fit,
                system_id="source_side_agency_components",
                outcome_id=outcome_id,
                equation_index=equation_index,
                coefficient_index=1,
                coefficient_id=f"beta_{outcome_id}",
                regressor_id=source_system_contract.regressor_ids[0],
                sample_hash=sample_hash,
                design_hash=source_design_hash,
                descriptive_only=True,
                covariance_contract=contract.covariance,
            )
        )
    within_outcomes = within_system_contract.outcome_ids
    within_equation_count = len(within_outcomes)
    if len(within_system_contract.coefficient_ids) != 2 * within_equation_count:
        raise _validation_failure(
            "common_sample_design_failed",
            "Within-system coefficient registry does not map two regressors per equation",
        )
    within_primary_coefficient_ids = (
        within_system_contract.coefficient_ids[:within_equation_count]
    )
    within_secondary_coefficient_ids = (
        within_system_contract.coefficient_ids[within_equation_count:]
    )
    for equation_index, outcome_id in enumerate(within_outcomes):
        for coefficient_index, coefficient_id, regressor_id in (
            (
                1,
                within_primary_coefficient_ids[equation_index],
                within_system_contract.regressor_ids[0],
            ),
            (
                2,
                within_secondary_coefficient_ids[equation_index],
                within_system_contract.regressor_ids[1],
            ),
        ):
            estimate_rows.append(
                _estimate_row(
                    system=within_aggregate_fit,
                    system_id=within_system_contract.system_id,
                    outcome_id=outcome_id,
                    equation_index=equation_index,
                    coefficient_index=coefficient_index,
                    coefficient_id=coefficient_id,
                    regressor_id=regressor_id,
                    sample_hash=sample_hash,
                    design_hash=within_design_hash,
                    descriptive_only=False,
                    covariance_contract=contract.covariance,
                )
            )
    for equation_index, outcome_id in enumerate(agency_component_ids):
        for coefficient_index, aggregate_ids, regressor_id in (
            (
                1,
                within_primary_coefficient_ids,
                within_system_contract.regressor_ids[0],
            ),
            (
                2,
                within_secondary_coefficient_ids,
                within_system_contract.regressor_ids[1],
            ),
        ):
            prefix = aggregate_ids[0].split("_", 1)[0]
            estimate_rows.append(
                _estimate_row(
                    system=within_component_fit,
                    system_id="within_bank_agency_components",
                    outcome_id=outcome_id,
                    equation_index=equation_index,
                    coefficient_index=coefficient_index,
                    coefficient_id=f"{prefix}_{outcome_id}",
                    regressor_id=regressor_id,
                    sample_hash=sample_hash,
                    design_hash=within_design_hash,
                    descriptive_only=True,
                    covariance_contract=contract.covariance,
                )
            )

    source_adding_formula = formula_by_id[
        "source_agency_coefficient_adding_up"
    ]
    within_adding_formula = formula_by_id[
        "within_agency_coefficient_adding_up"
    ]
    if (
        source_adding_formula.tolerance_kind
        != "relative_max_one_or_coefficients"
        or within_adding_formula.tolerance_kind
        != "relative_max_one_or_coefficients"
        or source_adding_formula.tolerance_scale_formula
        != "max(1, abs(aggregate_coefficient), abs(sum_component_coefficients))"
        or within_adding_formula.tolerance_scale_formula
        != "max(1, abs(aggregate_coefficient), abs(sum_component_coefficients))"
    ):
        raise _validation_failure(
            "coefficient_adding_up_failed",
            "Coefficient adding-up formula is not the implemented literal denominator",
        )
    source_adding_up = _relative_adding_up_error(
        source_aggregate_fit.equation_fits[1].coefficients[1],
        [
            fit.coefficients[1]
            for fit in source_component_fit.equation_fits
        ],
    )
    within_adding_up = _relative_adding_up_error(
        within_aggregate_fit.equation_fits[0].coefficients[1],
        [
            fit.coefficients[1]
            for fit in within_component_fit.equation_fits
        ],
    )
    adding_up_evidence = {
        "source_agency_beta_relative_error": source_adding_up,
        "within_agency_theta_relative_error": within_adding_up,
        "source_relative_tolerance": source_adding_formula.tolerance,
        "within_relative_tolerance": within_adding_formula.tolerance,
        "denominator": (
            "max(1, abs(aggregate_coefficient), "
            "abs(sum_component_coefficients))"
        ),
    }
    if (
        source_adding_formula.tolerance
        != contract.sample.coefficient_adding_up_relative_tolerance
        or within_adding_formula.tolerance
        != contract.sample.coefficient_adding_up_relative_tolerance
        or source_adding_up > source_adding_formula.tolerance
        or within_adding_up > within_adding_formula.tolerance
    ):
        raise _validation_failure(
            "coefficient_adding_up_failed",
            "Agency aggregate/component coefficients do not add up",
            **adding_up_evidence,
        )

    systems_by_id = {
        source_system_contract.system_id: source_system_contract,
        within_system_contract.system_id: within_system_contract,
    }
    fits_by_system_id = {
        source_system_contract.system_id: source_aggregate_fit,
        within_system_contract.system_id: within_aggregate_fit,
    }
    columns_by_system_id = {
        source_system_contract.system_id: source_columns,
        within_system_contract.system_id: within_columns,
    }
    hypotheses_by_id = {
        hypothesis.hypothesis_id: hypothesis
        for hypothesis in contract.wald_hypotheses
    }
    if (
        len(hypotheses_by_id) != len(contract.wald_hypotheses)
        or set(hypotheses_by_id) != set(contract.holm.hypothesis_ids)
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "Wald registry and Holm family do not align exactly",
        )
    ordered_hypotheses = tuple(
        hypotheses_by_id[hypothesis_id]
        for hypothesis_id in contract.holm.hypothesis_ids
    )
    wald_results_list: list[WaldTestResult] = []
    for hypothesis in ordered_hypotheses:
        if hypothesis.system_id not in fits_by_system_id:
            raise _validation_failure(
                "common_sample_design_failed",
                "Wald hypothesis names an unknown response system",
                hypothesis_id=hypothesis.hypothesis_id,
                system_id=hypothesis.system_id,
            )
        system_contract = systems_by_id[hypothesis.system_id]
        design_columns = columns_by_system_id[hypothesis.system_id]
        if (
            hypothesis.coefficient_ids
            != tuple(selection.coefficient_id for selection in hypothesis.selections)
            or hypothesis.degrees_of_freedom != len(hypothesis.selections)
        ):
            raise _validation_failure(
                "common_sample_design_failed",
                "Wald coefficient IDs, selections, and degrees of freedom drifted",
                hypothesis_id=hypothesis.hypothesis_id,
            )
        selections: list[CoefficientSelection] = []
        for selection in hypothesis.selections:
            if (
                selection.null_value != 0.0
                or not 0
                <= selection.equation_index
                < len(system_contract.outcome_ids)
                or system_contract.outcome_ids[selection.equation_index]
                != selection.outcome_id
                or not 0
                <= selection.coefficient_index
                < len(design_columns)
                or design_columns[selection.coefficient_index]
                != selection.regressor_id
            ):
                raise _validation_failure(
                    "common_sample_design_failed",
                    "A typed Wald selection does not map to its declared equation/design",
                    hypothesis_id=hypothesis.hypothesis_id,
                    selection=asdict(selection),
                )
            selections.append(
                CoefficientSelection(
                    selection.equation_index,
                    selection.coefficient_index,
                )
            )
        try:
            result = wald_zero_test(
                fits_by_system_id[hypothesis.system_id],
                selections,
            )
        except ValueError as exc:
            raise _validation_failure(
                "rank_gate_failed",
                "A declared Wald covariance is singular or invalid",
                hypothesis_id=hypothesis.hypothesis_id,
                error=str(exc),
            ) from exc
        wald_results_list.append(result)
    wald_results = tuple(wald_results_list)
    adjusted_p_values = holm_adjust_three(
        [result.p_value for result in wald_results]
    )
    wald_rows = [
        {
            "hypothesis_id": hypothesis.hypothesis_id,
            "coefficient_ids": hypothesis.coefficient_ids,
            "statistic": result.statistic,
            "degrees_of_freedom": result.degrees_of_freedom,
            "raw_p_value": result.p_value,
            "holm_adjusted_p_value": adjusted,
            "passed": adjusted <= contract.holm.familywise_alpha,
            "sample_hash": sample_hash,
            "covariance_estimator": contract.covariance.estimator,
            "kernel": contract.covariance.kernel,
            "hac_lags": contract.covariance.lag_quarters,
            "prewhitening": contract.covariance.prewhitening,
            "finite_sample_correction": (
                contract.covariance.finite_sample_correction
            ),
            "test_sidedness": contract.covariance.test_sidedness,
        }
            for hypothesis, result, adjusted in zip(
                ordered_hypotheses,
            wald_results,
            adjusted_p_values,
            strict=True,
        )
    ]

    estimate_by_coefficient_id = {
        str(row["coefficient_id"]): row for row in estimate_rows
    }
    if len(estimate_by_coefficient_id) != len(estimate_rows):
        raise _validation_failure(
            "common_sample_design_failed",
            "Estimate rows contain duplicate coefficient IDs",
        )
    selection_by_coefficient_id: dict[str, tuple[Any, Any]] = {}
    for hypothesis in ordered_hypotheses:
        for selection in hypothesis.selections:
            if selection.coefficient_id in selection_by_coefficient_id:
                raise _validation_failure(
                    "common_sample_design_failed",
                    "A coefficient occurs in multiple promotion hypotheses",
                    coefficient_id=selection.coefficient_id,
                )
            selection_by_coefficient_id[selection.coefficient_id] = (
                hypothesis,
                selection,
            )
    if (
        len(contract.influence_groups) != OPEN02_HOLM_FAMILY_SIZE
        or len(
            {group.group_id for group in contract.influence_groups}
        )
        != OPEN02_HOLM_FAMILY_SIZE
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "OPEN-02 requires exactly three unique influence groups",
        )
    full_groups: dict[
        str,
        tuple[list[float], list[float], tuple[str, ...]],
    ] = {}
    for group in contract.influence_groups:
        hypothesis = hypotheses_by_id.get(group.hypothesis_id)
        if (
            hypothesis is None
            or group.coefficient_ids != hypothesis.coefficient_ids
        ):
            raise _validation_failure(
                "common_sample_design_failed",
                "Influence group does not map exactly to its Wald hypothesis",
                group_id=group.group_id,
                hypothesis_id=group.hypothesis_id,
            )
        full_coefficients: list[float] = []
        full_p_values: list[float] = []
        for coefficient_id in group.coefficient_ids:
            mapped_hypothesis, selection = selection_by_coefficient_id[
                coefficient_id
            ]
            fit = fits_by_system_id[mapped_hypothesis.system_id]
            full_coefficients.append(
                fit.equation_fits[selection.equation_index].coefficients[
                    selection.coefficient_index
                ]
            )
            full_p_values.append(
                float(
                    estimate_by_coefficient_id[coefficient_id][
                        "raw_p_value"
                    ]
                )
            )
        full_groups[group.group_id] = (
            full_coefficients,
            full_p_values,
            group.coefficient_ids,
        )
    influence_refits: dict[str, dict[str, list[list[float]]]] = {
        group_id: {"leave_one": [], "leave_block": []}
        for group_id in full_groups
    }
    influence_rows: list[dict[str, Any]] = []
    deletion_source_adding_errors: list[float] = []
    deletion_within_adding_errors: list[float] = []

    expected_leave_one_fits = len(quarters)
    expected_leave_block_fits = (
        len(quarters) - contract.influence.block_deletion_size + 1
    )
    if (
        contract.influence.quarter_deletion_size != 1
        or contract.influence.quarter_deletion_fits != expected_leave_one_fits
        or contract.influence.block_deletion_size != 4
        or contract.influence.block_deletion_fits != expected_leave_block_fits
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "Influence contract does not define exactly 96 leave-one and 93 leave-four refits",
            quarter_deletion_size=contract.influence.quarter_deletion_size,
            quarter_deletion_fits=contract.influence.quarter_deletion_fits,
            block_deletion_size=contract.influence.block_deletion_size,
            block_deletion_fits=contract.influence.block_deletion_fits,
            expected_leave_one_fits=expected_leave_one_fits,
            expected_leave_block_fits=expected_leave_block_fits,
        )
    deletion_families = (
        (
            "leave_one",
            contract.influence.quarter_deletion_size,
            range(contract.influence.quarter_deletion_fits),
        ),
        (
            "leave_block",
            contract.influence.block_deletion_size,
            range(contract.influence.block_deletion_fits),
        ),
    )
    for deletion_kind, deletion_size, starts in deletion_families:
        for start in starts:
            stop = start + deletion_size
            if stop > len(quarters):
                raise _validation_failure(
                    "coverage_gate_failed",
                    "Influence contract requests an out-of-sample deletion",
                    deletion_kind=deletion_kind,
                    start=start,
                    stop=stop,
                )
            keep = [index for index in range(len(quarters)) if not start <= index < stop]
            source_refits = _refit_equations(
                [
                    [outcome[index] for index in keep]
                    for outcome in source_aggregate_outcomes
                ],
                [source_design[index] for index in keep],
            )
            within_refits = _refit_equations(
                [
                    [outcome[index] for index in keep]
                    for outcome in within_aggregate_outcomes
                ],
                [within_design[index] for index in keep],
            )
            source_component_refits = _refit_equations(
                [
                    [outcome[index] for index in keep]
                    for outcome in source_component_outcomes
                ],
                [source_design[index] for index in keep],
            )
            within_component_refits = _refit_equations(
                [
                    [outcome[index] for index in keep]
                    for outcome in within_component_outcomes
                ],
                [within_design[index] for index in keep],
            )
            deletion_source_adding_error = _relative_adding_up_error(
                source_refits[1].coefficients[1],
                [
                    fit.coefficients[1]
                    for fit in source_component_refits
                ],
            )
            deletion_within_adding_error = _relative_adding_up_error(
                within_refits[0].coefficients[1],
                [
                    fit.coefficients[1]
                    for fit in within_component_refits
                ],
            )
            deletion_source_adding_errors.append(
                deletion_source_adding_error
            )
            deletion_within_adding_errors.append(
                deletion_within_adding_error
            )
            if (
                deletion_source_adding_error
                > source_adding_formula.tolerance
                or deletion_within_adding_error
                > within_adding_formula.tolerance
            ):
                raise _validation_failure(
                    "coefficient_adding_up_failed",
                    "A deletion refit violates agency coefficient adding-up",
                    deletion_kind=deletion_kind,
                    omitted_quarters=quarters[start:stop],
                    source_relative_error=deletion_source_adding_error,
                    within_relative_error=deletion_within_adding_error,
                )
            refits_by_system_id = {
                source_system_contract.system_id: source_refits,
                within_system_contract.system_id: within_refits,
            }
            candidates: dict[str, list[float]] = {}
            for group in contract.influence_groups:
                candidate: list[float] = []
                for coefficient_id in group.coefficient_ids:
                    hypothesis, selection = selection_by_coefficient_id[
                        coefficient_id
                    ]
                    candidate.append(
                        refits_by_system_id[hypothesis.system_id][
                            selection.equation_index
                        ].coefficients[selection.coefficient_index]
                    )
                candidates[group.group_id] = candidate
            omitted = quarters[start:stop]
            for group_id, candidate in candidates.items():
                full_coefficients, full_p_values, coefficient_ids = full_groups[group_id]
                influence_refits[group_id][deletion_kind].append(candidate)
                significant_indices = {
                    index
                    for index, p_value in enumerate(full_p_values)
                    if p_value
                    <= contract.influence.sign_stability_raw_p_threshold
                }
                sign_flip = any(
                    _sign_reversed(full_coefficients[index], candidate[index])
                    for index in significant_indices
                )
                influence_rows.append(
                    {
                        "group_id": group_id,
                        "coefficient_ids": coefficient_ids,
                        "deletion_kind": deletion_kind,
                        "omitted_quarters": omitted,
                        "omitted_start": omitted[0],
                        "omitted_end": omitted[-1],
                        "refit_observations": len(keep),
                        "refit_coefficients": tuple(candidate),
                        "relative_l2_influence": relative_l2_influence(
                            full_coefficients,
                            candidate,
                            denominator_floor=(
                                contract.influence.relative_l2_denominator_floor
                            ),
                        ),
                        "sign_flip_for_significant_coefficient": sign_flip,
                    }
                )

    influence_summaries: list[dict[str, Any]] = []
    for group_id, (full_coefficients, full_p_values, coefficient_ids) in (
        full_groups.items()
    ):
        group_refits = influence_refits[group_id]
        gate = evaluate_influence_gate(
            full_coefficients,
            full_p_values,
            group_refits["leave_one"],
            group_refits["leave_block"],
            leave_one_threshold=contract.influence.maximum_quarter_influence,
            leave_block_threshold=contract.influence.maximum_block_influence,
            denominator_floor=contract.influence.relative_l2_denominator_floor,
            significance_alpha=contract.influence.sign_stability_raw_p_threshold,
        )
        group_rows = [
            row for row in influence_rows if row["group_id"] == group_id
        ]
        worst_one = max(
            (
                row
                for row in group_rows
                if row["deletion_kind"] == "leave_one"
            ),
            key=lambda row: row["relative_l2_influence"],
        )
        worst_block = max(
            (
                row
                for row in group_rows
                if row["deletion_kind"] == "leave_block"
            ),
            key=lambda row: row["relative_l2_influence"],
        )
        influence_summaries.append(
            {
                "group_id": group_id,
                "coefficient_ids": coefficient_ids,
                "full_coefficients": tuple(full_coefficients),
                "full_raw_p_values": tuple(full_p_values),
                "leave_one_fits": len(group_refits["leave_one"]),
                "leave_block_fits": len(group_refits["leave_block"]),
                "maximum_leave_one_influence": (
                    gate.maximum_leave_one_influence
                ),
                "maximum_leave_block_influence": (
                    gate.maximum_leave_block_influence
                ),
                "worst_leave_one_omission": worst_one["omitted_quarters"],
                "worst_leave_block_omission": worst_block["omitted_quarters"],
                "leave_one_passed": gate.leave_one_passed,
                "leave_block_passed": gate.leave_block_passed,
                "sign_flip_detected": gate.sign_flip_detected,
                "passed": gate.passed,
                "reason_codes": gate.reason_codes,
            }
        )

    adding_up_evidence.update(
        {
            "deletion_refit_policy": (
                contract.influence.refit_policy
            ),
            "deletion_fits_checked": (
                contract.influence.quarter_deletion_fits
                + contract.influence.block_deletion_fits
            ),
            "maximum_deletion_source_agency_beta_relative_error": max(
                deletion_source_adding_errors
            ),
            "maximum_deletion_within_agency_theta_relative_error": max(
                deletion_within_adding_errors
            ),
        }
    )

    eligibility = evaluate_open02_eligibility(
        wald_rows,
        influence_summaries,
        contract=contract,
    )
    source_column_hash = _quarter_hash(source_columns)
    within_column_hash = _quarter_hash(within_columns)
    source_equations = (
        *source_system_contract.outcome_ids,
        *source_system_contract.agency_component_outcome_ids,
    )
    within_equations = (
        *within_system_contract.outcome_ids,
        *within_system_contract.agency_component_outcome_ids,
    )
    equation_hashes = {
        source_system_contract.system_id: {
            outcome_id: {
                "row_hash": row_hash,
                "column_hash": source_column_hash,
                "design_hash": source_design_hash,
            }
            for outcome_id in source_equations
        },
        within_system_contract.system_id: {
            outcome_id: {
                "row_hash": row_hash,
                "column_hash": within_column_hash,
                "design_hash": within_design_hash,
            }
            for outcome_id in within_equations
        },
    }
    if contract.sample.required_common_hashes != (
        "row_hash",
        "column_hash",
        "design_hash",
    ) or any(
        len(
            {
                tuple(record[field] for field in contract.sample.required_common_hashes)
                for record in records.values()
            }
        )
        != 1
        for records in equation_hashes.values()
    ):
        raise _validation_failure(
            "common_sample_design_failed",
            "Relevant equations do not share exact row, column, and design hashes",
        )
    design_evidence = {
        "sample_hash": sample_hash,
        "row_hash": row_hash,
        "accepted_design_input_sha256": accepted_design_hash,
        "source": {
            "columns": source_columns,
            "column_hash": source_column_hash,
            "design_hash": source_design_hash,
            "equations": source_equations,
        },
        "within": {
            "columns": within_columns,
            "column_hash": within_column_hash,
            "design_hash": within_design_hash,
            "equations": within_equations,
        },
        "equation_hashes": equation_hashes,
    }
    gate_evidence: dict[str, Any] = {
        "metadata": {
            "series_count": source_evidence["series_count"],
            "series_keys": source_evidence["series_keys"],
        },
        "vintage": {
            "observation_vintage": observation_vintage,
            "cutoff": contract.sample.observation_vintage_cutoff,
            "accepted_tdcest_bundle_generated_at": (
                contract.sample.accepted_tdcest_bundle_generated_at
            ),
        },
        "coverage": {
            "observations": len(panel_rows),
            "start_quarter": quarters[0],
            "end_quarter": quarters[-1],
            "sample_hash": sample_hash,
        },
        **{
                gate_id: {
                    "maximum_absolute_residual": identity_residuals[gate_id],
                    "tolerance": identity_tolerances[gate_id],
            }
            for gate_id in (
                "treasury_component",
                "accepted_component_reconciliation",
                "leave_out_reconstruction",
                "us_agency_identity",
                "three_sector_agency_identity",
            )
        },
        "coefficient_adding_up": adding_up_evidence,
        "common_sample_design": design_evidence,
        "rank": {
            "source_parameters": len(source_columns),
            "within_parameters": len(within_columns),
            "observations": len(panel_rows),
            "full_rank": True,
        },
        "lineage": lineage_evidence,
    }
    declared_gate_ids = tuple(
        gate.gate_id for gate in contract.validity_gates
    )
    if set(declared_gate_ids) != set(gate_evidence):
        raise _validation_failure(
            "common_sample_design_failed",
            "Typed deterministic gates and generated evidence do not align",
            declared=list(declared_gate_ids),
            generated=list(gate_evidence),
        )
    deterministic_gates = tuple(
        {
            "gate_number": index,
            "gate_id": gate.gate_id,
            "reason_code": gate.reason_code,
            "requirement": gate.requirement,
            "passed": True,
            "evidence": gate_evidence[gate.gate_id],
        }
        for index, gate in enumerate(contract.validity_gates, start=1)
    )
    gate_map = {
        row["gate_id"]: {
            "passed": row["passed"],
            "reason_code": row["reason_code"],
            "requirement": row["requirement"],
            "evidence": row["evidence"],
        }
        for row in deterministic_gates
    }
    column_hashes = {
        "source": design_evidence["source"]["column_hash"],
        "within": design_evidence["within"]["column_hash"],
    }
    design_hashes = {
        "source": source_design_hash,
        "within": within_design_hash,
    }
    wald_statistics = {
        row["hypothesis_id"]: row["statistic"] for row in wald_rows
    }
    raw_p_values = {
        row["hypothesis_id"]: row["raw_p_value"] for row in wald_rows
    }
    holm_adjusted_p_values = {
        row["hypothesis_id"]: row["holm_adjusted_p_value"]
        for row in wald_rows
    }
    influence_maxima = {
        row["group_id"]: {
            "leave_one": row["maximum_leave_one_influence"],
            "leave_block": row["maximum_leave_block_influence"],
        }
        for row in influence_summaries
    }
    acceptance = {
        "open_id": "OPEN-02",
        **eligibility,
        "all_deterministic_gates_passed": True,
        "deterministic_gates": deterministic_gates,
        "gates": gate_map,
        "sample_hash": sample_hash,
        "row_hash": row_hash,
        "column_hashes": column_hashes,
        "design_hashes": design_hashes,
        "source_bundle_sha256": source_evidence["source_bundle_sha256"],
        "accepted_standardized_rows_sha256": standardized_hash,
        "accepted_design_rows_sha256": accepted_design_hash,
        "input_hashes": {
            "official_archive": contract.source.archive_sha256,
            "official_normalized_rows": source_evidence[
                "normalized_rows_sha256"
            ],
            "source_bundle": source_evidence["source_bundle_sha256"],
            "accepted_standardized_rows": standardized_hash,
            "accepted_design_rows": accepted_design_hash,
        },
        "panel_sha256": _stable_hash(panel_rows),
        "estimate_rows_sha256": _stable_hash(estimate_rows),
        "wald_rows_sha256": _stable_hash(wald_rows),
        "influence_rows_sha256": _stable_hash(influence_rows),
        "identity_evidence": identity_residuals,
        "identity_errors": identity_residuals,
        "coefficient_adding_up_evidence": adding_up_evidence,
        "design_evidence": design_evidence,
        "wald_statistics": wald_statistics,
        "raw_p_values": raw_p_values,
        "holm_adjusted_p_values": holm_adjusted_p_values,
        "influence_maxima": influence_maxima,
        "covariance_contract": asdict(contract.covariance),
        "observation_vintage": observation_vintage,
    }
    return Open02PipelineResult(
        panel_rows=tuple(panel_rows),
        estimate_rows=tuple(estimate_rows),
        wald_rows=tuple(wald_rows),
        influence_rows=tuple(influence_rows),
        influence_summaries=tuple(influence_summaries),
        acceptance=acceptance,
    )


__all__ = [
    "CoefficientSelection",
    "InfluenceGateResult",
    "OLSFit",
    "OPEN02_HAC_LAGS",
    "OPEN02_HOLM_FAMILY_SIZE",
    "OPEN02_INFLUENCE_DENOMINATOR_FLOOR",
    "OPEN02_LEAVE_BLOCK_INFLUENCE_THRESHOLD",
    "OPEN02_LEAVE_ONE_INFLUENCE_THRESHOLD",
    "OPEN02_SIGNIFICANCE_ALPHA",
    "Open02PipelineResult",
    "Open02ValidationError",
    "StackedSystemFit",
    "WaldTestResult",
    "chi_square_survival",
    "evaluate_open02_eligibility",
    "evaluate_influence_gate",
    "fit_ols",
    "fit_stacked_system",
    "holm_adjust_three",
    "relative_l2_influence",
    "run_open02_pipeline",
    "wald_zero_test",
]
