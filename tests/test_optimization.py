import pandas as pd
import pytest

from portfolio_risk_lab.optimization.minimum_variance import (
    OptimizationError,
    equal_weight_portfolio,
    minimum_variance_portfolio,
)


@pytest.fixture
def optimizer_inputs() -> tuple[pd.Series, pd.DataFrame]:
    expected = pd.Series([0.08, 0.12, 0.10], index=["A", "B", "C"])
    covariance = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.09, 0.01], [0.00, 0.01, 0.06]],
        index=expected.index,
        columns=expected.index,
    )
    return expected, covariance


def test_equal_weights_sum_to_one_and_respect_limit() -> None:
    weights = equal_weight_portfolio(["A", "B", "C"], max_weight=0.4)

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert (weights <= 0.4 + 1e-12).all()


def test_minimum_variance_is_long_only_and_respects_limit(
    optimizer_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    expected, covariance = optimizer_inputs

    result = minimum_variance_portfolio(expected, covariance, max_weight=0.5)

    assert result.weights.sum() == pytest.approx(1.0, abs=1e-7)
    assert (result.weights >= -1e-8).all()
    assert (result.weights <= 0.5 + 1e-7).all()
    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.diagnostics.weight_sum_residual <= result.diagnostics.constraint_tolerance
    assert result.diagnostics.lower_bound_violation <= result.diagnostics.constraint_tolerance
    assert result.diagnostics.upper_bound_violation <= result.diagnostics.constraint_tolerance
    assert result.diagnostics.solver


def test_target_return_constraint(optimizer_inputs: tuple[pd.Series, pd.DataFrame]) -> None:
    expected, covariance = optimizer_inputs

    result = minimum_variance_portfolio(
        expected,
        covariance,
        max_weight=0.8,
        target_return=0.105,
    )

    assert result.annual_return >= 0.105 - 1e-7
    assert result.diagnostics.target_return_residual is not None
    assert result.diagnostics.target_return_residual >= -result.diagnostics.constraint_tolerance


def test_infeasible_max_weight_is_not_silently_changed() -> None:
    expected = pd.Series([0.08, 0.12], index=["A", "B"])
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=expected.index, columns=expected.index
    )

    with pytest.raises(OptimizationError, match="约束不可行"):
        minimum_variance_portfolio(expected, covariance, max_weight=0.4)


def test_non_psd_covariance_is_rejected() -> None:
    expected = pd.Series([0.08, 0.12], index=["A", "B"])
    covariance = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]], index=expected.index, columns=expected.index
    )

    with pytest.raises(OptimizationError, match="半正定"):
        minimum_variance_portfolio(expected, covariance)


def test_short_selling_requires_explicit_lower_bound(
    optimizer_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    expected, covariance = optimizer_inputs

    with pytest.raises(OptimizationError, match="必须显式设置"):
        minimum_variance_portfolio(expected, covariance, allow_short=True)
