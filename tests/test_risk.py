import numpy as np
import pandas as pd
import pytest

from portfolio_risk_lab.analytics.risk import (
    covariance_matrix,
    historical_cvar,
    historical_var,
    portfolio_volatility,
    risk_contributions,
)


def test_historical_var_and_cvar_use_positive_loss_sign() -> None:
    returns = pd.Series([-0.10, -0.03, 0.00, 0.02, 0.04])
    expected_losses = -returns.to_numpy()
    expected_var = np.quantile(expected_losses, 0.8)
    expected_cvar = expected_losses[expected_losses >= expected_var].mean()

    assert historical_var(returns, 0.8) == pytest.approx(expected_var)
    assert historical_cvar(returns, 0.8) == pytest.approx(expected_cvar)
    assert historical_var(returns, 0.8) > 0


def test_covariance_uses_complete_rows_and_annualizes() -> None:
    returns = pd.DataFrame({"A": [0.1, 0.2, np.nan], "B": [0.2, 0.4, 0.1]})

    result = covariance_matrix(returns, periods_per_year=2)
    expected = returns.dropna().cov(ddof=1) * 2

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "returns",
    [
        pd.DataFrame({"A": [0.1], "B": [0.2]}),
        pd.DataFrame({"A": [np.nan, np.nan], "B": [0.1, 0.2]}),
        pd.DataFrame({"A": [0.1, 0.2], "B": ["bad", "data"]}),
    ],
)
def test_covariance_rejects_abnormal_input(returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        covariance_matrix(returns)


def test_risk_contributions_sum_to_portfolio_volatility() -> None:
    covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    weights = pd.Series([0.6, 0.4], index=["A", "B"])

    volatility = portfolio_volatility(weights, covariance)
    contributions = risk_contributions(weights, covariance)

    assert contributions["component_contribution"].sum() == pytest.approx(volatility)
    assert contributions["percentage_contribution"].sum() == pytest.approx(1.0)
