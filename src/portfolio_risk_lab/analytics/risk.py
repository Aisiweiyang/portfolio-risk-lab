"""历史模拟风险、协方差与组合风险分解。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from portfolio_risk_lab.analytics.returns import _validate_returns


def covariance_matrix(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """使用完整共同观测计算样本协方差并年化。"""
    clean = complete_return_frame(returns)
    if len(clean) < 2:
        raise ValueError("协方差矩阵至少需要 2 个完整共同收益观测。")
    return clean.cov(ddof=1) * periods_per_year


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """使用完整共同观测计算 Pearson 相关矩阵。"""
    clean = complete_return_frame(returns)
    if len(clean) < 2:
        raise ValueError("相关矩阵至少需要 2 个完整共同收益观测。")
    return clean.corr()


def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """历史损失分布的置信分位数；正数代表损失。"""
    losses = _losses(returns, confidence_level)
    return float(np.quantile(losses, confidence_level))


def historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """损失不小于 VaR 的样本平均值（Expected Shortfall）。"""
    losses = _losses(returns, confidence_level)
    value_at_risk = float(np.quantile(losses, confidence_level))
    tail = losses[losses >= value_at_risk]
    return float(np.mean(tail))


def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """计算 sqrt(w'Σw)；协方差应已采用目标周期口径。"""
    aligned = _aligned_weights(weights, covariance)
    variance = float(aligned @ covariance.to_numpy(dtype=float) @ aligned)
    if variance < -1e-12:
        raise ValueError("组合方差为负，协方差矩阵可能无效。")
    return math.sqrt(max(variance, 0.0))


def risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """Euler 分解：各资产对组合波动率的贡献之和等于组合波动率。"""
    aligned = _aligned_weights(weights, covariance)
    sigma = portfolio_volatility(weights, covariance)
    if np.isclose(sigma, 0.0):
        raise ValueError("零波动组合无法计算风险贡献。")
    marginal = covariance.to_numpy(dtype=float) @ aligned / sigma
    component = aligned * marginal
    return pd.DataFrame(
        {
            "weight": aligned,
            "marginal_contribution": marginal,
            "component_contribution": component,
            "percentage_contribution": component / sigma,
        },
        index=covariance.index,
    )


def complete_return_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """返回所有资产均有有限收益的共同样本，不修改原始输入。"""
    _validate_returns(returns)
    if returns.shape[1] < 1:
        raise ValueError("至少需要一个资产收益列。")
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if clean.empty:
        raise ValueError("没有完整的共同收益观测。")
    return clean.astype(float)


def _losses(returns: pd.Series, confidence_level: float) -> np.ndarray:
    _validate_returns(returns)
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("置信水平必须在 0.5 与 1 之间。")
    clean = returns.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        raise ValueError("VaR/CVaR 至少需要一个有效收益观测。")
    return -clean


def _aligned_weights(weights: pd.Series, covariance: pd.DataFrame) -> np.ndarray:
    if covariance.empty or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("协方差矩阵必须是非空方阵。")
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("协方差矩阵的行列标签必须一致且顺序相同。")
    if set(weights.index) != set(covariance.index):
        raise ValueError("权重资产必须与协方差矩阵资产完全一致。")
    aligned = weights.reindex(covariance.index).to_numpy(dtype=float)
    if not np.isfinite(aligned).all():
        raise ValueError("权重必须是有限数值。")
    return aligned
