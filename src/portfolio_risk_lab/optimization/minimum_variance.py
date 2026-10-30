"""等权与长仓最小方差组合。"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_risk_lab.analytics.risk import portfolio_volatility


class OptimizationError(RuntimeError):
    """优化约束、输入或求解状态无效。"""


@dataclass(frozen=True)
class OptimizationDiagnostics:
    """求解器与约束残差，便于识别数值上不可靠的结果。"""

    solver: str
    objective_value: float
    solve_time_seconds: float | None
    weight_sum_residual: float
    lower_bound_violation: float
    upper_bound_violation: float
    target_return_residual: float | None
    constraint_tolerance: float


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    annual_return: float
    annual_volatility: float
    status: str
    diagnostics: OptimizationDiagnostics


def equal_weight_portfolio(assets: list[str], max_weight: float = 1.0) -> pd.Series:
    """构造等权组合，并严格检查用户最大权重约束。"""
    if not assets or len(set(assets)) != len(assets):
        raise OptimizationError("资产列表不能为空且不得重复。")
    _validate_max_weight(len(assets), max_weight)
    return pd.Series(1.0 / len(assets), index=assets, name="weight")


def minimum_variance_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    max_weight: float = 1.0,
    allow_short: bool = False,
    min_weight: float | None = None,
    target_return: float | None = None,
) -> OptimizationResult:
    """用 CVXPY 求解最小方差，可选目标年化收益约束。"""
    assets, mu, sigma = _validate_inputs(expected_returns, covariance)
    if not allow_short:
        lower_bound = 0.0
    else:
        if min_weight is None:
            raise OptimizationError("允许做空时必须显式设置单项资产最小权重。")
        lower_bound = min_weight
    _validate_weight_bounds(len(assets), lower_bound, max_weight)
    if target_return is not None and not np.isfinite(target_return):
        raise OptimizationError("目标收益率必须是有限数值。")

    # CVXPY/SciPy 在 Windows 冷启动时加载成本较高，只在真正求解时导入。
    import cvxpy as cp

    weights = cp.Variable(len(assets))
    constraints: list[cp.Constraint] = [
        cp.sum(weights) == 1,
        weights >= lower_bound,
        weights <= max_weight,
    ]
    if target_return is not None:
        constraints.append(mu @ weights >= target_return)

    problem = cp.Problem(cp.Minimize(cp.quad_form(weights, sigma)), constraints)
    installed = set(cp.installed_solvers())
    solver = "CLARABEL" if "CLARABEL" in installed else "SCS" if "SCS" in installed else None
    if solver is None:
        raise OptimizationError("未找到可用的 CVXPY 求解器（需要 CLARABEL 或 SCS）。")
    try:
        problem.solve(solver=solver)
    except cp.error.SolverError as exc:
        raise OptimizationError(f"求解器 {solver} 失败：{exc}") from exc

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        reason = "约束可能不可行" if "infeasible" in problem.status else "问题可能无界或数值不稳定"
        raise OptimizationError(f"优化失败，求解状态：{problem.status}；{reason}。")

    solved = np.asarray(weights.value, dtype=float)
    tolerance = 1e-6 if problem.status == cp.OPTIMAL else 1e-4
    weight_sum_residual = abs(float(solved.sum()) - 1.0)
    lower_bound_violation = max(lower_bound - float(solved.min()), 0.0)
    upper_bound_violation = max(float(solved.max()) - max_weight, 0.0)
    realized_return = float(mu @ solved)
    target_return_residual = None if target_return is None else realized_return - target_return
    violations = [weight_sum_residual, lower_bound_violation, upper_bound_violation]
    if target_return_residual is not None:
        violations.append(max(-target_return_residual, 0.0))
    if max(violations) > tolerance:
        raise OptimizationError(
            "求解器返回的权重未满足约束容差："
            f"最大残差 {max(violations):.3e}，允许 {tolerance:.3e}。"
        )

    result_weights = pd.Series(solved, index=assets, name="weight")
    solver_name = str(problem.solver_stats.solver_name or solver)
    solve_time = problem.solver_stats.solve_time
    return OptimizationResult(
        weights=result_weights,
        annual_return=realized_return,
        annual_volatility=portfolio_volatility(result_weights, covariance),
        status=str(problem.status),
        diagnostics=OptimizationDiagnostics(
            solver=solver_name,
            objective_value=float(problem.value),
            solve_time_seconds=None if solve_time is None else float(solve_time),
            weight_sum_residual=weight_sum_residual,
            lower_bound_violation=lower_bound_violation,
            upper_bound_violation=upper_bound_violation,
            target_return_residual=target_return_residual,
            constraint_tolerance=tolerance,
        ),
    )


def _validate_inputs(
    expected_returns: pd.Series, covariance: pd.DataFrame
) -> tuple[list[Hashable], np.ndarray, np.ndarray]:
    if expected_returns.empty:
        raise OptimizationError("预期收益不能为空。")
    if covariance.empty or covariance.shape[0] != covariance.shape[1]:
        raise OptimizationError("协方差矩阵必须是非空方阵。")
    if list(covariance.index) != list(covariance.columns):
        raise OptimizationError("协方差矩阵行列标签必须一致。")
    if covariance.index.has_duplicates or expected_returns.index.has_duplicates:
        raise OptimizationError("优化输入的资产标签不得重复。")
    if set(expected_returns.index) != set(covariance.index):
        raise OptimizationError("预期收益与协方差矩阵的资产集合必须一致。")
    assets = list(covariance.index)
    mu = expected_returns.reindex(covariance.index).to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise OptimizationError("优化输入包含缺失值或无穷值。")
    if not np.allclose(sigma, sigma.T, rtol=1e-8, atol=1e-10):
        raise OptimizationError("协方差矩阵必须对称。")
    minimum_eigenvalue = float(np.linalg.eigvalsh(sigma).min())
    if minimum_eigenvalue < -1e-9:
        raise OptimizationError("协方差矩阵不是半正定矩阵。")
    return assets, mu, sigma


def _validate_weight_bounds(
    asset_count: int,
    min_weight: float,
    max_weight: float,
) -> None:
    if not np.isfinite(min_weight) or not np.isfinite(max_weight):
        raise OptimizationError("单项资产权重上下限必须是有限数值。")
    if min_weight >= max_weight:
        raise OptimizationError("单项资产最小权重必须小于最大权重。")
    if asset_count * max_weight < 1.0 - 1e-12:
        raise OptimizationError(
            f"约束不可行：{asset_count} 个资产 × 最大权重 {max_weight:.2%} 小于 100%。"
        )
    if asset_count * min_weight > 1.0 + 1e-12:
        raise OptimizationError(
            f"约束不可行：{asset_count} 个资产 × 最小权重 {min_weight:.2%} 大于 100%。"
        )


def _validate_max_weight(asset_count: int, max_weight: float) -> None:
    """兼容等权组合的长仓上限检查。"""
    _validate_weight_bounds(asset_count, 0.0, max_weight)
