"""回撤计算。"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import pandas as pd

from portfolio_risk_lab.analytics.returns import _validate_returns


@dataclass(frozen=True)
class DrawdownResult:
    maximum_drawdown: float
    peak_date: Hashable | None
    trough_date: Hashable
    peak_is_initial_value: bool


def drawdown_series(returns: pd.Series) -> pd.Series:
    """以简单收益率生成回撤序列；回撤为非正数。"""
    _validate_returns(returns)
    clean = returns.dropna()
    wealth = (1.0 + clean).cumprod()
    running_peak = wealth.cummax().clip(lower=1.0)
    return wealth / running_peak - 1.0


def maximum_drawdown(returns: pd.Series) -> DrawdownResult:
    """返回最大回撤及峰值—谷值区间。

    当样本从第一期便开始下跌时，峰值是收益序列之前的初始净值 1，
    此时 peak_date 为 None，避免伪造一个实际交易日期。
    """
    clean = returns.dropna()
    drawdowns = drawdown_series(clean)
    trough_date = drawdowns.idxmin()
    wealth = (1.0 + clean).cumprod()
    before_trough = wealth.loc[:trough_date]
    peak_is_initial_value = float(before_trough.max()) < 1.0
    peak_date = None if peak_is_initial_value else before_trough.idxmax()
    return DrawdownResult(
        maximum_drawdown=float(drawdowns.loc[trough_date]),
        peak_date=peak_date,
        trough_date=trough_date,
        peak_is_initial_value=peak_is_initial_value,
    )
