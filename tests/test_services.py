import pandas as pd
import pytest

from portfolio_risk_lab.services.analysis import prepare_portfolio_estimates


def test_portfolio_estimates_use_identical_complete_sample() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, None],
            "B": [0.03, -0.01, 1.00],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
    )

    estimates = prepare_portfolio_estimates(returns, periods_per_year=2)

    assert estimates.observation_count == 2
    pd.testing.assert_frame_equal(estimates.aligned_returns, returns.iloc[:2])
    expected_b_growth = (1.03 * 0.99) - 1.0
    assert estimates.expected_returns["B"] == pytest.approx(expected_b_growth)
    expected_covariance = returns.iloc[:2].cov(ddof=1) * 2
    pd.testing.assert_frame_equal(estimates.covariance, expected_covariance)


def test_portfolio_estimates_require_two_common_observations() -> None:
    returns = pd.DataFrame({"A": [0.01, None], "B": [0.02, 0.03]})

    with pytest.raises(ValueError, match="至少需要 2"):
        prepare_portfolio_estimates(returns)
