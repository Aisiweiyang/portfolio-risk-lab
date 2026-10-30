import numpy as np
import pandas as pd
import pytest

from portfolio_risk_lab.analytics.drawdown import maximum_drawdown
from portfolio_risk_lab.analytics.performance import annualized_return, annualized_volatility
from portfolio_risk_lab.analytics.returns import cumulative_returns, log_returns, simple_returns


def test_known_price_simple_log_and_cumulative_returns() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=index)

    simple = simple_returns(prices)
    logarithmic = log_returns(prices)
    cumulative = cumulative_returns(simple)

    np.testing.assert_allclose(simple["A"].to_numpy(), [0.1, -0.1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        logarithmic["A"].to_numpy(),
        [np.log(1.1), np.log(0.9)],
        rtol=1e-12,
        atol=1e-12,
    )
    assert cumulative["A"].iloc[-1] == pytest.approx(-0.01)


def test_annualized_return_and_volatility_known_values() -> None:
    returns = pd.Series([0.1, -0.1])

    assert annualized_return(returns, periods_per_year=2) == pytest.approx(-0.01)
    assert annualized_volatility(returns, periods_per_year=2) == pytest.approx(0.2)


def test_maximum_drawdown_and_interval() -> None:
    index = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    returns = pd.Series([0.1, -0.2, 0.1], index=index)

    result = maximum_drawdown(returns)

    assert result.maximum_drawdown == pytest.approx(-0.2)
    assert result.peak_date == pd.Timestamp("2025-01-02")
    assert result.trough_date == pd.Timestamp("2025-01-03")
    assert result.peak_is_initial_value is False


def test_drawdown_reports_virtual_initial_peak_without_fake_date() -> None:
    index = pd.to_datetime(["2025-01-02", "2025-01-03"])
    returns = pd.Series([-0.1, 0.02], index=index)

    result = maximum_drawdown(returns)

    assert result.maximum_drawdown == pytest.approx(-0.1)
    assert result.peak_date is None
    assert result.trough_date == pd.Timestamp("2025-01-02")
    assert result.peak_is_initial_value is True


def test_simple_returns_never_implicitly_forward_fills() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, np.nan, 110.0]}, index=index)

    result = simple_returns(prices)

    assert result["A"].isna().all()
