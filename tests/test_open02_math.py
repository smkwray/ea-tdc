from __future__ import annotations

from math import isclose, sqrt

import pytest

from ea_tdc.estimation import _ols
from ea_tdc.open02 import (
    CoefficientSelection,
    OLSFit,
    StackedSystemFit,
    chi_square_survival,
    evaluate_influence_gate,
    fit_ols,
    fit_stacked_system,
    holm_adjust_three,
    relative_l2_influence,
    wald_zero_test,
)


def _design(observations: int = 16) -> list[list[float]]:
    return [[1.0, float(index)] for index in range(observations)]


def _outcome(
    intercept: float,
    slope: float,
    residual_pattern: list[float],
) -> list[float]:
    return [
        intercept + slope * index + residual_pattern[index % len(residual_pattern)]
        for index in range(16)
    ]


def test_fit_ols_recovers_coefficients_and_retains_residuals() -> None:
    x_rows = _design()
    y_values = [2.0 + 3.0 * row[1] for row in x_rows]

    fit = fit_ols(y_values, x_rows)

    assert fit.coefficients == pytest.approx((2.0, 3.0))
    assert fit.fitted_values == pytest.approx(y_values)
    assert fit.residuals == pytest.approx([0.0] * len(y_values), abs=1e-12)
    assert fit.observations == 16
    assert fit.parameters == 2


def test_fit_ols_fails_closed_on_singular_or_misaligned_design() -> None:
    singular = [[1.0, 2.0] for _ in range(8)]
    with pytest.raises(ValueError, match="not full rank"):
        fit_ols([float(index) for index in range(8)], singular)
    with pytest.raises(ValueError, match="identical observation counts"):
        fit_ols([1.0, 2.0], _design())


def test_stacked_hac_is_symmetric_and_has_equation_major_dimensions() -> None:
    x_rows = _design()
    first = _outcome(1.0, 0.5, [1.0, -0.5, 0.25, -0.75])
    second = _outcome(-2.0, -0.25, [-0.5, 1.25, -0.25, 0.5])

    fit = fit_stacked_system([first, second], x_rows)

    assert fit.equation_count == 2
    assert fit.parameters_per_equation == 2
    assert fit.hac_lags == 4
    assert fit.kernel == "bartlett"
    assert fit.prewhitened is False
    assert fit.finite_sample_scale == pytest.approx(16 / 14)
    assert len(fit.covariance) == 4
    assert all(len(row) == 4 for row in fit.covariance)
    for row in range(4):
        for column in range(4):
            assert fit.covariance[row][column] == pytest.approx(
                fit.covariance[column][row],
                abs=1e-15,
            )
    assert fit.flat_coefficients == pytest.approx(
        (
            *fit.equation_fits[0].coefficients,
            *fit.equation_fits[1].coefficients,
        )
    )
    assert fit.flat_index(1, 0) == 2


def test_single_equation_stacked_hac_matches_existing_newey_west_block() -> None:
    x_rows = _design()
    y_values = _outcome(4.0, 0.75, [1.0, -1.0, 0.5, -0.25])

    stacked = fit_stacked_system([y_values], x_rows)
    existing = _ols(
        y_values,
        x_rows,
        covariance_estimator="newey_west",
        covariance_lags=4,
    )

    for row in range(2):
        for column in range(2):
            assert stacked.covariance[row][column] == pytest.approx(
                existing.covariance[row][column],
                rel=1e-12,
                abs=1e-14,
            )


def test_cross_equation_hac_blocks_preserve_residual_scaling() -> None:
    x_rows = _design()
    first = _outcome(1.0, 0.5, [1.0, -0.5, 0.25, -0.75])
    second = [2.0 * value for value in first]

    fit = fit_stacked_system([first, second], x_rows)

    for row in range(2):
        for column in range(2):
            base = fit.covariance[row][column]
            assert fit.covariance[row][2 + column] == pytest.approx(
                2.0 * base,
                rel=1e-12,
                abs=1e-14,
            )
            assert fit.covariance[2 + row][2 + column] == pytest.approx(
                4.0 * base,
                rel=1e-12,
                abs=1e-14,
            )


@pytest.mark.parametrize(
    ("degrees_of_freedom", "critical_value"),
    [
        (1, 3.841458820694124),
        (2, 5.991464547107979),
        (3, 7.814727903251179),
        (4, 9.487729036781154),
    ],
)
def test_chi_square_survival_matches_five_percent_critical_values(
    degrees_of_freedom: int,
    critical_value: float,
) -> None:
    assert chi_square_survival(
        critical_value,
        degrees_of_freedom,
    ) == pytest.approx(0.05, rel=2e-10, abs=2e-12)


def test_chi_square_survival_handles_zero_and_extreme_tail() -> None:
    assert chi_square_survival(0.0, 3) == 1.0
    assert 0.0 <= chi_square_survival(1000.0, 3) < 1e-100


def test_wald_zero_test_uses_declared_system_coefficient_selection() -> None:
    equation_fit = OLSFit(
        coefficients=(0.0, sqrt(3.841458820694124)),
        fitted_values=(0.0,),
        residuals=(0.0,),
        observations=10,
        parameters=2,
    )
    system = StackedSystemFit(
        equation_fits=(equation_fit,),
        covariance=((2.0, 0.0), (0.0, 1.0)),
        observations=10,
        parameters_per_equation=2,
        hac_lags=4,
        kernel="bartlett",
        prewhitened=False,
        finite_sample_scale=1.25,
    )

    result = wald_zero_test(
        system,
        [CoefficientSelection(0, 1)],
    )

    assert result.statistic == pytest.approx(3.841458820694124)
    assert result.degrees_of_freedom == 1
    assert result.p_value == pytest.approx(0.05, rel=2e-10)


def test_wald_zero_test_rejects_duplicate_or_singular_selection() -> None:
    equation_fit = OLSFit(
        coefficients=(1.0, 2.0),
        fitted_values=(0.0,),
        residuals=(0.0,),
        observations=10,
        parameters=2,
    )
    system = StackedSystemFit(
        equation_fits=(equation_fit,),
        covariance=((1.0, 0.0), (0.0, 0.0)),
        observations=10,
        parameters_per_equation=2,
        hac_lags=4,
        kernel="bartlett",
        prewhitened=False,
        finite_sample_scale=1.25,
    )
    selection = CoefficientSelection(0, 1)
    with pytest.raises(ValueError, match="unique"):
        wald_zero_test(system, [selection, selection])
    with pytest.raises(ValueError, match="singular"):
        wald_zero_test(system, [selection])


def test_holm_adjusts_exactly_three_p_values_in_original_order() -> None:
    assert holm_adjust_three([0.04, 0.01, 0.03]) == pytest.approx(
        (0.06, 0.03, 0.06)
    )
    with pytest.raises(ValueError, match="exactly three"):
        holm_adjust_three([0.01, 0.02])


def test_relative_l2_influence_uses_norm_and_frozen_denominator_floor() -> None:
    assert relative_l2_influence([3.0, 4.0], [3.0, 5.0]) == pytest.approx(
        0.2
    )
    assert relative_l2_influence([0.0], [1e-12]) == pytest.approx(1.0)


def test_influence_thresholds_are_inclusive_at_exact_boundaries() -> None:
    result = evaluate_influence_gate(
        [1.0],
        [0.10],
        leave_one_refits=[[1.25]],
        leave_block_refits=[[1.50]],
    )

    assert result.maximum_leave_one_influence == pytest.approx(0.25)
    assert result.maximum_leave_block_influence == pytest.approx(0.50)
    assert result.leave_one_passed is True
    assert result.leave_block_passed is True
    assert result.passed is True
    assert result.reason_codes == ()


def test_influence_gate_reports_threshold_and_sign_flip_reasons() -> None:
    result = evaluate_influence_gate(
        [1.0, -2.0],
        [0.05, 0.051],
        leave_one_refits=[[-0.1, -2.0]],
        leave_block_refits=[[1.0, -3.5]],
    )

    assert result.sign_flip_detected is True
    assert result.passed is False
    assert result.reason_codes == (
        "leave_quarter_influence_gt_0_25",
        "leave_block_influence_gt_0_50",
        "sign_flip_under_influence",
    )


def test_non_significant_coefficient_sign_change_is_not_a_gate_failure() -> None:
    result = evaluate_influence_gate(
        [1.0],
        [0.051],
        leave_one_refits=[[-0.1]],
        leave_block_refits=[[1.0]],
        leave_one_threshold=2.0,
    )

    assert result.sign_flip_detected is False
    assert result.passed is True
