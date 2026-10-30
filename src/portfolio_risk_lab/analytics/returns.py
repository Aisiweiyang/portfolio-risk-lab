"""价格与收益率转换。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_risk_lab.data.validator import validate_price_frame


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """计算简单收益率 P_t / P_(t-1) - 1，不填充缺失价格。"""
    validate_price_frame(prices)
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """计算对数收益率 ln(P_t / P_(t-1))，不填充缺失价格。"""
    validate_price_frame(prices)
    return np.log(prices / prices.shift(1)).iloc[1:]


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """计算从 1 元初始财富得到的累计简单收益率。"""
    _validate_returns(returns)
    return (1.0 + returns).cumprod(skipna=False) - 1.0


def _validate_returns(returns: pd.Series | pd.DataFrame) -> None:
    if returns.empty:
        raise ValueError("收益率序列为空。")
    numeric = (
        returns.select_dtypes(include="number") if isinstance(returns, pd.DataFrame) else returns
    )
    if isinstance(returns, pd.DataFrame) and numeric.shape[1] != returns.shape[1]:
        raise ValueError("收益率必须全部为数值列。")
    values = numeric.to_numpy(dtype=float)
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        raise ValueError("收益率没有可用观测。")
    if np.isinf(finite).any():
        raise ValueError("收益率包含无穷值。")
    if (finite < -1.0).any():
        raise ValueError("简单收益率不能小于 -100%。")
