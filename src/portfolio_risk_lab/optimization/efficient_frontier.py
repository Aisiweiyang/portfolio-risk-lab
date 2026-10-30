"""长仓、全额投资和单项权重上限下的有效前沿。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_risk_lab.optimization.minimum_variance import (
    OptimizationError,
    OptimizationResult,
    equal_weight_portfolio,
    minimum_variance_portfolio,
)


@dataclass(frozen=True)
class FeasibleReturnRange:
    """权重约束下可达到的最小与最大年化预期收益。"""

    minimum_return: float
    maximum_return: float
    minimum_return_weights: pd.Series
    maximum_return_weights: pd.Series


@dataclass(frozen=True)
class FrontierPoint:
    """一个目标收益率及其对应的最小方差解。"""

    target_return: float
    result: OptimizationResult


@dataclass(frozen=True)
class EfficientFrontierResult:
    """完整有效前沿及约束下的收益可行区间。"""

    points: tuple[FrontierPoint, ...]
    feasible_range: FeasibleReturnRange

    def metrics_frame(self) -> pd.DataFrame:
        """返回适合展示与导出的前沿指标表。"""
        return pd.DataFrame(
            [
                {
                    "target_return": point.target_return,
                    "annual_return": point.result.annual_return,
                    "annual_volatility": point.result.annual_volatility,
                    "status": point.result.status,
                    "solver": point.result.diagnostics.solver,
                }
                for point in self.points
            ]
        )

    def weights_frame(self) -> pd.DataFrame:
        """返回每个前沿点的资产权重，行索引是目标收益率。"""
        frame = pd.DataFrame(
            [point.result.weights for point in self.points],
            index=[point.target_return for point in self.points],
        )
        frame.index.name = "target_return"
        return frame


def feasible_return_range(
    expected_returns: pd.Series,
    *,
    max_weight: float = 1.0,
) -> FeasibleReturnRange:
    """用确定性贪心分配计算长仓盒约束下的收益极值。"""
    _validate_expected_returns(expected_returns)
    equal_weight_portfolio(list(expected_returns.index), max_weight=max_weight)
    minimum_weights = _extreme_return_weights(
        expected_returns,
        max_weight=max_weight,
        ascending=True,
    )
    maximum_weights = _extreme_return_weights(
        expected_returns,
        max_weight=max_weight,
        ascending=False,
    )
    return FeasibleReturnRange(
        minimum_return=float(expected_returns @ minimum_weights),
        maximum_return=float(expected_returns @ maximum_weights),
        minimum_return_weights=minimum_weights,
        maximum_return_weights=maximum_weights,
    )


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    max_weight: float = 1.0,
    point_count: int = 12,
) -> EfficientFrontierResult:
    """生成从全局最小方差点到最大可行收益点的有效前沿。

    当前函数仅支持禁止做空；每个前沿点都严格使用调用方给出的同一
    预期收益向量、协方差矩阵和最大权重约束。
    """
    if not isinstance(point_count, int) or not 2 <= point_count <= 100:
        raise OptimizationError("有效前沿点数必须是 2 到 100 之间的整数。")

    return_range = feasible_return_range(expected_returns, max_weight=max_weight)
    minimum = minimum_variance_portfolio(
        expected_returns,
        covariance,
        max_weight=max_weight,
        allow_short=False,
    )
    start_return = minimum.annual_return
    end_return = return_range.maximum_return
    if end_return < start_return - 1e-8:
        raise OptimizationError("最大可行收益低于最小方差组合收益，输入估计可能不一致。")

    if np.isclose(start_return, end_return, rtol=1e-9, atol=1e-10):
        return EfficientFrontierResult(
            points=(FrontierPoint(target_return=start_return, result=minimum),),
            feasible_range=return_range,
        )

    targets = np.linspace(start_return, end_return, point_count)
    points: list[FrontierPoint] = [FrontierPoint(target_return=float(targets[0]), result=minimum)]
    for target in targets[1:]:
        try:
            result = minimum_variance_portfolio(
                expected_returns,
                covariance,
                max_weight=max_weight,
                allow_short=False,
                target_return=float(target),
            )
        except OptimizationError as exc:
            raise OptimizationError(f"有效前沿在目标收益率 {target:.2%} 处求解失败：{exc}") from exc
        points.append(FrontierPoint(target_return=float(target), result=result))

    return EfficientFrontierResult(points=tuple(points), feasible_range=return_range)


def _validate_expected_returns(expected_returns: pd.Series) -> None:
    if expected_returns.empty:
        raise OptimizationError("预期收益不能为空。")
    if expected_returns.index.has_duplicates:
        raise OptimizationError("预期收益的资产标签不得重复。")
    values = expected_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise OptimizationError("预期收益包含缺失值或无穷值。")


def _extreme_return_weights(
    expected_returns: pd.Series,
    *,
    max_weight: float,
    ascending: bool,
) -> pd.Series:
    ordered = expected_returns.sort_values(
        ascending=ascending,
        kind="stable",
    )
    remaining = 1.0
    weights = pd.Series(0.0, index=expected_returns.index, name="weight")
    for asset in ordered.index:
        allocation = min(max_weight, remaining)
        weights.loc[asset] = allocation
        remaining -= allocation
        if remaining <= 1e-12:
            break
    if remaining > 1e-10:
        raise OptimizationError("权重上限约束不可行，无法完成全额投资。")
    return weights
