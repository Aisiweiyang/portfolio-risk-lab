"""滚动组合回测引擎。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from portfolio_risk_lab.analytics.drawdown import drawdown_series
from portfolio_risk_lab.analytics.performance import performance_summary
from portfolio_risk_lab.backtest.costs import proportional_transaction_cost
from portfolio_risk_lab.optimization.minimum_variance import (
    OptimizationError,
    equal_weight_portfolio,
    minimum_variance_portfolio,
)
from portfolio_risk_lab.services.analysis import prepare_portfolio_estimates

BacktestStrategy = Literal["equal_weight", "minimum_variance"]


class BacktestError(RuntimeError):
    """回测输入、时序或求解失败。"""


class BacktestConfig(BaseModel):
    """经校验的滚动回测参数。"""

    model_config = ConfigDict(frozen=True)

    strategy: BacktestStrategy = "equal_weight"
    training_window: int = Field(default=20, ge=2)
    rebalance_frequency: int = Field(default=5, ge=1)
    transaction_cost_rate: float = Field(default=0.001, ge=0.0, lt=0.5)
    max_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    periods_per_year: int = Field(default=252, ge=2, le=366)


@dataclass(frozen=True)
class BacktestResult:
    """策略、基准及逐期审计数据。"""

    net_returns: pd.DataFrame
    net_value: pd.DataFrame
    drawdowns: pd.DataFrame
    target_weights: pd.DataFrame
    close_weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    rebalance_log: pd.DataFrame
    config: BacktestConfig

    def performance(
        self,
        *,
        annual_risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """返回策略与基准的统一口径绩效表。"""
        return performance_summary(
            self.net_returns,
            periods_per_year=self.config.periods_per_year,
            annual_risk_free_rate=annual_risk_free_rate,
        )

    def export_frame(self) -> pd.DataFrame:
        """组合逐期结果，适合 CSV 导出。"""
        return pd.concat(
            {
                "return": self.net_returns,
                "net_value": self.net_value,
                "drawdown": self.drawdowns,
                "close_weight": self.close_weights,
                "audit": pd.concat(
                    [
                        self.turnover.rename("turnover"),
                        self.transaction_costs.rename("transaction_cost"),
                    ],
                    axis=1,
                ),
            },
            axis=1,
        )


@dataclass(frozen=True)
class _SimulationResult:
    net_returns: pd.Series
    net_value: pd.Series
    close_weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series


def run_rolling_backtest(
    returns: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """运行严格按时间推进的滚动回测。

    决策位置 d 使用 [d-training_window+1, d] 的收益，生成的目标权重只从
    d+1 位置的收益开始生效。任何窗口求解失败都会终止并报告对应决策日。
    """
    clean = _validate_backtest_returns(returns, config)
    assets = list(clean.columns)
    try:
        benchmark_weights = equal_weight_portfolio(assets)
        equal_weight_portfolio(assets, max_weight=config.max_weight)
    except OptimizationError as exc:
        raise BacktestError(str(exc)) from exc

    target_rows: list[pd.Series] = []
    log_rows: list[dict[str, object]] = []
    for decision_position in range(
        config.training_window - 1,
        len(clean) - 1,
        config.rebalance_frequency,
    ):
        window_start = decision_position - config.training_window + 1
        training = clean.iloc[window_start : decision_position + 1]
        decision_date = clean.index[decision_position]
        application_date = clean.index[decision_position + 1]
        try:
            target, solver, status = _target_weights(training, config)
        except (OptimizationError, ValueError) as exc:
            raise BacktestError(f"决策日 {decision_date:%Y-%m-%d} 回测求解失败：{exc}") from exc
        target.name = application_date
        target_rows.append(target)
        log_rows.append(
            {
                "training_start": training.index[0],
                "training_end": training.index[-1],
                "decision_date": decision_date,
                "application_date": application_date,
                "training_observations": len(training),
                "solver": solver,
                "status": status,
            }
        )

    target_weights = pd.DataFrame(target_rows)
    target_weights.index.name = "application_date"
    holding_returns = clean.iloc[config.training_window :]
    strategy = _simulate_targets(
        holding_returns,
        target_weights,
        config.transaction_cost_rate,
        name="strategy",
    )
    benchmark_targets = pd.DataFrame(
        [benchmark_weights],
        index=pd.DatetimeIndex([holding_returns.index[0]], name="application_date"),
    )
    benchmark = _simulate_targets(
        holding_returns,
        benchmark_targets,
        config.transaction_cost_rate,
        name="buy_and_hold",
    )

    net_returns = pd.concat([strategy.net_returns, benchmark.net_returns], axis=1)
    net_value = pd.concat([strategy.net_value, benchmark.net_value], axis=1)
    drawdowns = pd.DataFrame(
        {column: drawdown_series(net_returns[column]) for column in net_returns.columns}
    )
    return BacktestResult(
        net_returns=net_returns,
        net_value=net_value,
        drawdowns=drawdowns,
        target_weights=target_weights,
        close_weights=strategy.close_weights,
        turnover=strategy.turnover,
        transaction_costs=strategy.transaction_costs,
        rebalance_log=pd.DataFrame(log_rows),
        config=config,
    )


def _target_weights(
    training_returns: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.Series, str, str]:
    assets = list(training_returns.columns)
    if config.strategy == "equal_weight":
        return (
            equal_weight_portfolio(assets, max_weight=config.max_weight),
            "无需求解器",
            "deterministic",
        )
    estimates = prepare_portfolio_estimates(
        training_returns,
        periods_per_year=config.periods_per_year,
    )
    result = minimum_variance_portfolio(
        estimates.expected_returns,
        estimates.covariance,
        max_weight=config.max_weight,
        allow_short=False,
    )
    return result.weights, result.diagnostics.solver, result.status


def _simulate_targets(
    holding_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost_rate: float,
    *,
    name: str,
) -> _SimulationResult:
    assets = list(holding_returns.columns)
    current_weights = pd.Series(0.0, index=assets, dtype=float)
    net_returns: list[float] = []
    net_values: list[float] = []
    close_weights: list[pd.Series] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    net_value = 1.0

    for date, asset_returns in holding_returns.iterrows():
        turnover = 0.0
        cost_fraction = 0.0
        if date in target_weights.index:
            target = target_weights.loc[date].reindex(assets).astype(float)
            cost = proportional_transaction_cost(
                current_weights,
                target,
                transaction_cost_rate,
            )
            turnover = cost.traded_notional_ratio
            cost_fraction = cost.cost_fraction
            if cost_fraction >= 1.0:
                raise BacktestError(f"{date:%Y-%m-%d} 的交易成本不小于组合净值。")
            current_weights = target

        if not np.isclose(float(current_weights.sum()), 1.0, atol=1e-6):
            raise BacktestError(f"{date:%Y-%m-%d} 生效权重之和不等于 1。")
        gross_growth = float(
            current_weights.to_numpy(dtype=float) @ (1.0 + asset_returns.to_numpy(dtype=float))
        )
        if gross_growth <= 0.0:
            raise BacktestError(f"{date:%Y-%m-%d} 组合价值归零，无法继续回测。")
        period_growth = (1.0 - cost_fraction) * gross_growth
        period_return = period_growth - 1.0
        net_value *= period_growth
        current_weights = current_weights * (1.0 + asset_returns) / gross_growth

        net_returns.append(period_return)
        net_values.append(net_value)
        close_weights.append(current_weights.rename(date))
        turnover_values.append(turnover)
        cost_values.append(cost_fraction)

    index = holding_returns.index
    return _SimulationResult(
        net_returns=pd.Series(net_returns, index=index, name=name),
        net_value=pd.Series(net_values, index=index, name=name),
        close_weights=pd.DataFrame(close_weights, index=index),
        turnover=pd.Series(turnover_values, index=index, name="turnover"),
        transaction_costs=pd.Series(
            cost_values,
            index=index,
            name="transaction_cost",
        ),
    )


def _validate_backtest_returns(
    returns: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    if returns.empty or returns.shape[1] < 1:
        raise BacktestError("回测至少需要一个资产收益列。")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise BacktestError("回测收益必须使用 DatetimeIndex。")
    if returns.index.has_duplicates:
        raise BacktestError("回测日期不得重复。")
    if not returns.index.is_monotonic_increasing:
        raise BacktestError("回测日期必须严格升序。")
    if returns.columns.has_duplicates:
        raise BacktestError("回测资产列不得重复。")
    if returns.select_dtypes(include="number").shape[1] != returns.shape[1]:
        raise BacktestError("回测收益必须全部为数值列。")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BacktestError("回测收益包含缺失值或无穷值；请先明确完成交易日对齐。")
    if (values < -1.0).any():
        raise BacktestError("简单收益率不能小于 -100%。")
    if len(returns) - config.training_window < 2:
        raise BacktestError("训练窗口之后至少需要 2 个样本外收益观测。")
    return returns.astype(float).copy()
