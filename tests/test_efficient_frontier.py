import importlib

import numpy as np
import pandas as pd
import pytest

from portfolio_risk_lab.optimization.efficient_frontier import (
    efficient_frontier,
    feasible_return_range,
)
from portfolio_risk_lab.optimization.minimum_variance import OptimizationError

frontier_module = importlib.import_module("portfolio_risk_lab.optimization.efficient_frontier")


@pytest.fixture
def frontier_inputs() -> tuple[pd.Series, pd.DataFrame]:
    expected = pd.Series([0.05, 0.10, 0.15], index=["A", "B", "C"])
    covariance = pd.DataFrame(
        [[0.04, 0.006, 0.004], [0.006, 0.06, 0.008], [0.004, 0.008, 0.10]],
        index=expected.index,
        columns=expected.index,
    )
    return expected, covariance


def test_feasible_return_range_respects_maximum_weight() -> None:
    expected = pd.Series([0.05, 0.10, 0.15], index=["A", "B", "C"])

    result = feasible_return_range(expected, max_weight=0.6)

    assert result.minimum_return == pytest.approx(0.07)
    assert result.maximum_return == pytest.approx(0.13)
    assert result.minimum_return_weights.sum() == pytest.approx(1.0)
    assert result.maximum_return_weights.sum() == pytest.approx(1.0)
    assert result.minimum_return_weights.max() <= 0.6
    assert result.maximum_return_weights.max() <= 0.6


def test_efficient_frontier_is_ordered_and_respects_constraints(
    frontier_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    expected, covariance = frontier_inputs

    result = efficient_frontier(
        expected,
        covariance,
        max_weight=0.8,
        point_count=6,
    )

    metrics = result.metrics_frame()
    weights = result.weights_frame()
    assert len(metrics) == 6
    assert np.all(np.diff(metrics["target_return"]) > 0)
    assert np.all(np.diff(metrics["annual_return"]) >= -1e-7)
    assert np.all(np.diff(metrics["annual_volatility"]) >= -1e-7)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert weights.to_numpy().min() >= -1e-6
    assert weights.to_numpy().max() <= 0.8 + 1e-6
    assert metrics.iloc[-1]["annual_return"] == pytest.approx(
        result.feasible_range.maximum_return,
        abs=1e-6,
    )


@pytest.mark.parametrize("point_count", [1, 101, 2.5])
def test_frontier_rejects_invalid_point_count(
    frontier_inputs: tuple[pd.Series, pd.DataFrame],
    point_count: int,
) -> None:
    expected, covariance = frontier_inputs

    with pytest.raises(OptimizationError, match="点数"):
        efficient_frontier(expected, covariance, point_count=point_count)


def test_frontier_rejects_infeasible_maximum_weight(
    frontier_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    expected, covariance = frontier_inputs

    with pytest.raises(OptimizationError, match="约束不可行"):
        efficient_frontier(expected, covariance, max_weight=0.3)


def test_frontier_propagates_target_solver_failure(
    frontier_inputs: tuple[pd.Series, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, covariance = frontier_inputs
    real_solver = frontier_module.minimum_variance_portfolio

    def fail_target(*args: object, **kwargs: object):
        if kwargs.get("target_return") is not None:
            raise OptimizationError("固定测试故障")
        return real_solver(*args, **kwargs)

    monkeypatch.setattr(frontier_module, "minimum_variance_portfolio", fail_target)

    with pytest.raises(OptimizationError, match="有效前沿.*求解失败"):
        efficient_frontier(expected, covariance, point_count=3)
