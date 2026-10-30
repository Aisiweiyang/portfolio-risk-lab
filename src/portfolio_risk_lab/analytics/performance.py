"""年化绩效指标。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from portfolio_risk_lab.analytics.drawdown import maximum_drawdown
from portfolio_risk_lab.analytics.returns import _validate_returns


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """按实际有效观测数计算几何年化收益。"""
    clean = _clean_series(returns, periods_per_year, minimum=1)
    growth = float((1.0 + clean).prod())
    if growth == 0.0:
        return -1.0
    return growth ** (periods_per_year / len(clean)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """样本标准差乘以 sqrt(年化周期数)。"""
    clean = _clean_series(returns, periods_per_year, minimum=2)
    return float(clean.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> float:
    """几何年化超额收益除以年化波动率。"""
    volatility = annualized_volatility(returns, periods_per_year)
    if np.isclose(volatility, 0.0):
        return math.nan
    return (annualized_return(returns, periods_per_year) - annual_risk_free_rate) / volatility


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> float:
    """几何年化超额收益除以年化下行偏差。"""
    clean = _clean_series(returns, periods_per_year, minimum=1)
    daily_rf = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    downside = np.minimum(clean.to_numpy(dtype=float) - daily_rf, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year))
    if np.isclose(downside_deviation, 0.0):
        return math.nan
    return (annualized_return(clean, periods_per_year) - annual_risk_free_rate) / downside_deviation


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """几何年化收益除以最大回撤绝对值。"""
    drawdown = maximum_drawdown(returns).maximum_drawdown
    if np.isclose(drawdown, 0.0):
        return math.nan
    return annualized_return(returns, periods_per_year) / abs(drawdown)


def performance_summary(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """逐资产输出统一口径的绩效表。"""
    rows: dict[str, dict[str, float]] = {}
    for column in returns.columns:
        series = returns[column]
        rows[str(column)] = {
            "annual_return": annualized_return(series, periods_per_year),
            "annual_volatility": annualized_volatility(series, periods_per_year),
            "sharpe": sharpe_ratio(series, periods_per_year, annual_risk_free_rate),
            "sortino": sortino_ratio(series, periods_per_year, annual_risk_free_rate),
            "max_drawdown": maximum_drawdown(series).maximum_drawdown,
            "calmar": calmar_ratio(series, periods_per_year),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _clean_series(returns: pd.Series, periods_per_year: int, minimum: int) -> pd.Series:
    _validate_returns(returns)
    if periods_per_year < 2:
        raise ValueError("年化周期数必须至少为 2。")
    clean = returns.dropna().astype(float)
    if len(clean) < minimum:
        raise ValueError(f"至少需要 {minimum} 个有效收益率观测。")
    return clean
