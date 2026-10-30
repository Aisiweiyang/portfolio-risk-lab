"""协调核心金融计算，确保组合估计使用完全相同的样本。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from portfolio_risk_lab.analytics.performance import annualized_return
from portfolio_risk_lab.analytics.risk import complete_return_frame, covariance_matrix


@dataclass(frozen=True)
class PortfolioEstimates:
    """组合优化所需的统一样本和年化估计。"""

    aligned_returns: pd.DataFrame
    expected_returns: pd.Series
    covariance: pd.DataFrame

    @property
    def observation_count(self) -> int:
        return len(self.aligned_returns)


def prepare_portfolio_estimates(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> PortfolioEstimates:
    """使用共同完整收益日同时估计预期收益与协方差。"""
    aligned = complete_return_frame(returns)
    if len(aligned) < 2:
        raise ValueError("组合估计至少需要 2 个所有资产共同完整的收益观测。")
    expected = pd.Series(
        {
            column: annualized_return(aligned[column], periods_per_year)
            for column in aligned.columns
        },
        name="annual_expected_return",
        dtype=float,
    )
    covariance = covariance_matrix(aligned, periods_per_year)
    return PortfolioEstimates(
        aligned_returns=aligned.copy(),
        expected_returns=expected,
        covariance=covariance,
    )
