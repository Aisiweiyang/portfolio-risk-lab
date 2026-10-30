import numpy as np
import pandas as pd
import pytest

from portfolio_risk_lab.backtest.costs import proportional_transaction_cost
from portfolio_risk_lab.backtest.engine import (
    BacktestConfig,
    BacktestError,
    run_rolling_backtest,
)


@pytest.fixture
def fixed_returns() -> pd.DataFrame:
    index = pd.date_range("2025-01-02", periods=10, freq="B")
    return pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, 0.03, -0.02, 0.01, 0.00, 0.02, -0.01, 0.01],
            "B": [0.00, 0.02, -0.01, 0.01, 0.01, -0.02, 0.02, 0.00, 0.01, -0.01],
        },
        index=index,
    )


def test_transaction_cost_uses_traded_notional() -> None:
    previous = pd.Series([0.6, 0.4], index=["A", "B"])
    target = pd.Series([0.4, 0.6], index=["A", "B"])

    result = proportional_transaction_cost(previous, target, rate=0.001)

    assert result.traded_notional_ratio == pytest.approx(0.4)
    assert result.cost_fraction == pytest.approx(0.0004)


def test_first_weight_application_is_after_training_window(
    fixed_returns: pd.DataFrame,
) -> None:
    result = run_rolling_backtest(
        fixed_returns,
        BacktestConfig(training_window=3, rebalance_frequency=2),
    )

    first = result.rebalance_log.iloc[0]
    assert first["training_start"] == fixed_returns.index[0]
    assert first["training_end"] == fixed_returns.index[2]
    assert first["decision_date"] == fixed_returns.index[2]
    assert first["application_date"] == fixed_returns.index[3]
    assert result.net_returns.index[0] == fixed_returns.index[3]
    assert list(result.target_weights.index) == list(fixed_returns.index[[3, 5, 7, 9]])
    assert (result.rebalance_log["application_date"] > result.rebalance_log["decision_date"]).all()


def test_fixed_cost_is_deducted_before_period_return(
    fixed_returns: pd.DataFrame,
) -> None:
    cost_rate = 0.01
    result = run_rolling_backtest(
        fixed_returns,
        BacktestConfig(
            training_window=3,
            rebalance_frequency=2,
            transaction_cost_rate=cost_rate,
        ),
    )

    first_asset_return = fixed_returns.iloc[3]
    first_gross_return = float(first_asset_return.mean())
    expected_net_return = (1.0 - cost_rate) * (1.0 + first_gross_return) - 1.0
    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert result.transaction_costs.iloc[0] == pytest.approx(cost_rate)
    assert result.net_returns["strategy"].iloc[0] == pytest.approx(expected_net_return)
    np.testing.assert_allclose(
        result.transaction_costs,
        result.turnover * cost_rate,
    )


def test_cost_reduces_terminal_value(fixed_returns: pd.DataFrame) -> None:
    no_cost = run_rolling_backtest(
        fixed_returns,
        BacktestConfig(training_window=3, rebalance_frequency=2, transaction_cost_rate=0.0),
    )
    with_cost = run_rolling_backtest(
        fixed_returns,
        BacktestConfig(training_window=3, rebalance_frequency=2, transaction_cost_rate=0.01),
    )

    assert with_cost.net_value["strategy"].iloc[-1] < no_cost.net_value["strategy"].iloc[-1]
    assert with_cost.net_value["buy_and_hold"].iloc[-1] < no_cost.net_value["buy_and_hold"].iloc[-1]


def test_minimum_variance_weights_do_not_use_future_returns(
    fixed_returns: pd.DataFrame,
) -> None:
    config = BacktestConfig(
        strategy="minimum_variance",
        training_window=4,
        rebalance_frequency=2,
        max_weight=0.8,
    )
    baseline = run_rolling_backtest(fixed_returns, config)
    changed_future = fixed_returns.copy()
    changed_future.iloc[8:] = np.array([[0.40, -0.30], [-0.35, 0.45]])
    changed = run_rolling_backtest(changed_future, config)
    unaffected_dates = fixed_returns.index[[4, 6, 8]]

    np.testing.assert_allclose(baseline.target_weights.sum(axis=1), 1.0, atol=1e-7)
    assert (baseline.target_weights >= -1e-7).all().all()
    assert (baseline.target_weights <= 0.8 + 1e-7).all().all()
    pd.testing.assert_frame_equal(
        baseline.target_weights.loc[unaffected_dates],
        changed.target_weights.loc[unaffected_dates],
        rtol=1e-7,
        atol=1e-8,
    )


def test_buy_and_hold_is_not_rebalanced_after_initial_purchase(
    fixed_returns: pd.DataFrame,
) -> None:
    result = run_rolling_backtest(
        fixed_returns,
        BacktestConfig(training_window=3, rebalance_frequency=1),
    )

    first_holding_date = result.net_returns.index[0]
    expected_first_benchmark = float(fixed_returns.loc[first_holding_date].mean())
    assert result.net_returns.loc[first_holding_date, "buy_and_hold"] == pytest.approx(
        expected_first_benchmark - 0.001 * (1.0 + expected_first_benchmark)
    )
    assert not np.allclose(
        result.close_weights.iloc[-1].to_numpy(),
        [0.5, 0.5],
    )


def test_infeasible_weight_limit_is_reported(
    fixed_returns: pd.DataFrame,
) -> None:
    with pytest.raises(BacktestError, match="约束不可行"):
        run_rolling_backtest(
            fixed_returns,
            BacktestConfig(training_window=3, max_weight=0.4),
        )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.sort_index(ascending=False), "严格升序"),
        (
            lambda frame: frame.assign(A=frame["A"].mask(frame.index == frame.index[2])),
            "缺失值",
        ),
    ],
)
def test_invalid_backtest_input_is_rejected(
    fixed_returns: pd.DataFrame,
    mutator,
    message: str,
) -> None:
    with pytest.raises(BacktestError, match=message):
        run_rolling_backtest(
            mutator(fixed_returns),
            BacktestConfig(training_window=3),
        )
