"""Streamlit 各功能页，只协调展示，不实现核心金融公式。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from portfolio_risk_lab.analytics.drawdown import drawdown_series
from portfolio_risk_lab.analytics.performance import performance_summary
from portfolio_risk_lab.analytics.returns import cumulative_returns, log_returns
from portfolio_risk_lab.analytics.risk import (
    correlation_matrix,
    historical_cvar,
    historical_var,
    portfolio_volatility,
    risk_contributions,
)
from portfolio_risk_lab.backtest.engine import (
    BacktestConfig,
    BacktestError,
    run_rolling_backtest,
)
from portfolio_risk_lab.config import AnalysisConfig
from portfolio_risk_lab.data.loader import LoadedPriceData
from portfolio_risk_lab.optimization.efficient_frontier import (
    EfficientFrontierResult,
    efficient_frontier,
    feasible_return_range,
)
from portfolio_risk_lab.optimization.minimum_variance import (
    OptimizationError,
    equal_weight_portfolio,
    minimum_variance_portfolio,
)
from portfolio_risk_lab.services.analysis import prepare_portfolio_estimates
from portfolio_risk_lab.ui.charts import (
    backtest_drawdown_chart,
    backtest_net_value_chart,
    backtest_turnover_chart,
    correlation_heatmap,
    cumulative_return_chart,
    drawdown_chart,
    efficient_frontier_chart,
    portfolio_weights_chart,
    rolling_weights_chart,
)


def render_data_quality(loaded: LoadedPriceData) -> None:
    columns = st.columns(5)
    columns[0].metric("资产数", loaded.prices.shape[1])
    columns[1].metric("原始日期数", loaded.original_rows)
    columns[2].metric("处理后日期数", loaded.output_rows)
    columns[3].metric("处理前缺失", loaded.missing_before)
    columns[4].metric("处理后缺失", loaded.missing_after)
    st.write(
        f"数据区间：{loaded.prices.index.min():%Y-%m-%d} 至 {loaded.prices.index.max():%Y-%m-%d}"
    )
    st.dataframe(loaded.prices.tail(20), width="stretch")
    st.download_button(
        "下载清洗后的价格 CSV",
        loaded.prices.to_csv().encode("utf-8-sig"),
        file_name="cleaned_prices.csv",
        mime="text/csv",
    )


def render_analytics(
    loaded: LoadedPriceData,
    returns: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    summary = performance_summary(
        returns,
        config.periods_per_year,
        config.annual_risk_free_rate,
    )
    st.subheader("资产绩效")
    st.dataframe(_format_performance(summary), width="stretch")

    cumulative = cumulative_returns(returns)
    st.plotly_chart(cumulative_return_chart(cumulative), width="stretch")

    correlation = correlation_matrix(returns)
    st.plotly_chart(correlation_heatmap(correlation), width="stretch")

    selected = st.selectbox("查看单资产风险", list(returns.columns))
    asset_returns = returns[selected].dropna()
    var_value = historical_var(asset_returns, config.var_confidence_level)
    cvar_value = historical_cvar(asset_returns, config.var_confidence_level)
    drawdowns = drawdown_series(asset_returns)
    risk_columns = st.columns(3)
    confidence = config.var_confidence_level
    risk_columns[0].metric(f"{confidence:.0%} 单周期 VaR", f"{var_value:.2%}")
    risk_columns[1].metric(f"{confidence:.0%} 单周期 CVaR", f"{cvar_value:.2%}")
    risk_columns[2].metric("最大回撤", f"{drawdowns.min():.2%}")
    st.plotly_chart(drawdown_chart(drawdowns, str(selected)), width="stretch")

    with st.expander("对数收益与年化协方差"):
        st.dataframe(log_returns(loaded.prices).tail(10), width="stretch")
        estimates = prepare_portfolio_estimates(returns, config.periods_per_year)
        st.caption(f"协方差采用 {estimates.observation_count} 个共同完整收益观测。")
        st.dataframe(estimates.covariance, width="stretch")


def render_portfolio(returns: pd.DataFrame, config: AnalysisConfig) -> None:
    max_weight = st.slider("单项资产最大权重", 0.10, 1.00, 1.00, 0.05)
    mode = st.radio(
        "优化模式",
        ["最小方差", "目标收益率最小方差", "有效前沿"],
        horizontal=True,
    )
    try:
        estimates = prepare_portfolio_estimates(returns, config.periods_per_year)
        return_range = feasible_return_range(
            estimates.expected_returns,
            max_weight=max_weight,
        )
    except (OptimizationError, ValueError) as exc:
        st.error(str(exc))
        return

    st.caption(
        f"共同完整收益观测：{estimates.observation_count}；"
        f"当前权重约束下可行预期收益区间："
        f"{return_range.minimum_return:.2%} 至 {return_range.maximum_return:.2%}。"
    )
    st.caption(
        "当前收益向量采用逐资产历史几何年化收益；组合展示值为该向量的权重加权值，"
        "并非组合实际几何年化收益或未来收益预测。"
    )
    target_return: float | None = None
    frontier_point_count = 12
    if mode == "目标收益率最小方差":
        target_return = _target_return_input(
            return_range.minimum_return,
            return_range.maximum_return,
        )
    elif mode == "有效前沿":
        frontier_point_count = st.slider("有效前沿点数", 5, 30, 12, 1)

    if not st.button("运行组合优化", type="primary"):
        st.info("设置优化模式和约束后点击“运行组合优化”；求解只在明确请求时执行。")
        return

    frontier_result: EfficientFrontierResult | None = None
    try:
        equal_weights = equal_weight_portfolio(list(returns.columns), max_weight=max_weight)
        if mode == "有效前沿":
            frontier_result = efficient_frontier(
                estimates.expected_returns,
                estimates.covariance,
                max_weight=max_weight,
                point_count=frontier_point_count,
            )
            optimized = frontier_result.points[0].result
            optimized_label = "全局最小方差组合"
        else:
            optimized = minimum_variance_portfolio(
                estimates.expected_returns,
                estimates.covariance,
                max_weight=max_weight,
                allow_short=False,
                target_return=target_return,
            )
            optimized_label = "目标收益率组合" if target_return is not None else "最小方差组合"
    except (OptimizationError, ValueError) as exc:
        st.error(str(exc))
        return

    equal_return = float(estimates.expected_returns @ equal_weights)
    equal_volatility = portfolio_volatility(equal_weights, estimates.covariance)
    comparison = pd.DataFrame(
        {
            "等权组合": [equal_return, equal_volatility],
            optimized_label: [optimized.annual_return, optimized.annual_volatility],
        },
        index=["年化收益", "年化波动率"],
    )
    st.dataframe(comparison.map(lambda value: f"{value:.2%}"), width="stretch")

    weights = pd.concat(
        [equal_weights.rename("等权组合"), optimized.weights.rename(optimized_label)],
        axis=1,
    )
    st.plotly_chart(portfolio_weights_chart(weights), width="stretch")

    if frontier_result is not None:
        _render_frontier(frontier_result)

    diagnostics = optimized.diagnostics
    if optimized.status == "optimal_inaccurate":
        st.warning("求解器返回 optimal_inaccurate，请结合约束残差谨慎解释结果。")
    st.caption(f"求解器：{diagnostics.solver}；状态：{optimized.status}")
    with st.expander("求解诊断"):
        diagnostic_values = {
            "目标函数值": diagnostics.objective_value,
            "求解时间（秒）": diagnostics.solve_time_seconds,
            "权重和残差": diagnostics.weight_sum_residual,
            "下限违反量": diagnostics.lower_bound_violation,
            "上限违反量": diagnostics.upper_bound_violation,
            "约束容差": diagnostics.constraint_tolerance,
        }
        if diagnostics.target_return_residual is not None:
            diagnostic_values["目标收益率余量"] = diagnostics.target_return_residual
        diagnostic_frame = pd.Series(diagnostic_values, name="数值")
        st.dataframe(diagnostic_frame.to_frame(), width="stretch")

    contributions = risk_contributions(optimized.weights, estimates.covariance)
    st.subheader(f"{optimized_label}风险贡献")
    st.dataframe(contributions, width="stretch")
    st.download_button(
        "下载组合权重 CSV",
        weights.to_csv().encode("utf-8-sig"),
        file_name="portfolio_weights.csv",
        mime="text/csv",
    )


def render_backtest(returns: pd.DataFrame, config: AnalysisConfig) -> None:
    """滚动回测交互页；输入必须是已对齐的共同完整收益。"""
    st.caption(
        f"回测使用 {len(returns)} 个共同完整收益观测，"
        f"区间为 {returns.index.min():%Y-%m-%d} 至 {returns.index.max():%Y-%m-%d}。"
    )
    if len(returns) < 4:
        st.warning("滚动回测至少需要 4 个共同完整收益观测。")
        return
    strategy_label = st.selectbox(
        "回测策略",
        ["等权定期再平衡", "最小方差"],
    )
    strategy = {
        "等权定期再平衡": "equal_weight",
        "最小方差": "minimum_variance",
    }[strategy_label]
    maximum_training_window = len(returns) - 2
    default_training_window = min(20, maximum_training_window)
    training_window = st.number_input(
        "训练窗口（收益观测数）",
        min_value=2,
        max_value=maximum_training_window,
        value=default_training_window,
        step=1,
    )
    rebalance_frequency = st.number_input(
        "再平衡频率（持有期数）",
        min_value=1,
        max_value=len(returns),
        value=min(5, len(returns)),
        step=1,
    )
    transaction_cost_bps = st.number_input(
        "单边交易成本（基点/成交名义金额）",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        format="%.1f",
    )
    max_weight = st.slider(
        "回测单项资产最大权重",
        0.10,
        1.00,
        1.00,
        0.05,
    )
    st.caption(
        "决策日只使用截至当日的训练数据，权重从下一期开始生效；"
        "成本率乘以买卖分别计入的成交名义金额比例。"
    )
    if not st.button("运行滚动回测", type="primary"):
        st.info("设置参数后点击“运行滚动回测”。")
        return

    try:
        backtest_config = BacktestConfig(
            strategy=strategy,
            training_window=int(training_window),
            rebalance_frequency=int(rebalance_frequency),
            transaction_cost_rate=float(transaction_cost_bps) / 10_000.0,
            max_weight=float(max_weight),
            periods_per_year=config.periods_per_year,
        )
        result = run_rolling_backtest(returns, backtest_config)
    except (BacktestError, ValueError) as exc:
        st.error(str(exc))
        return

    labels = {
        "strategy": strategy_label,
        "buy_and_hold": "等权买入持有基准",
    }
    net_value = result.net_value.rename(columns=labels)
    drawdowns = result.drawdowns.rename(columns=labels)
    performance = result.performance(annual_risk_free_rate=config.annual_risk_free_rate).rename(
        index=labels
    )

    st.subheader("样本外回测结果")
    st.dataframe(_format_performance(performance), width="stretch")
    st.plotly_chart(backtest_net_value_chart(net_value), width="stretch")
    st.plotly_chart(backtest_drawdown_chart(drawdowns), width="stretch")

    metric_columns = st.columns(3)
    metric_columns[0].metric("再平衡次数", len(result.target_weights))
    metric_columns[1].metric("成交名义金额比例合计", f"{result.turnover.sum():.2%}")
    metric_columns[2].metric("调仓成本比例合计", f"{result.transaction_costs.sum():.3%}")
    st.plotly_chart(backtest_turnover_chart(result.turnover), width="stretch")
    st.plotly_chart(rolling_weights_chart(result.close_weights), width="stretch")

    with st.expander("回测时序审计"):
        st.dataframe(result.rebalance_log, width="stretch")
        st.caption("每行 application_date 必须严格晚于 decision_date。")
    st.download_button(
        "下载回测结果 CSV",
        result.export_frame().to_csv().encode("utf-8-sig"),
        file_name="rolling_backtest.csv",
        mime="text/csv",
    )


def render_help() -> None:
    st.markdown(
        """
        - **交易周期**：日频通常使用 252；周频、月频需要相应修改。
        - **无风险利率**：有效年利率；Sharpe/Sortino 使用同一年度口径。
        - **VaR/CVaR**：基于单期历史收益，正数代表损失，未预测未来。
        - **最小方差**：权重和为 1、默认禁止做空，并严格遵守最大权重。
        - **目标收益率组合**：在达到给定预期收益下寻找最低方差组合。
        - **有效前沿**：从全局最小方差点延伸到最大可行收益点，不跳过失败点。
        - **共同样本**：组合预期收益与协方差使用完全相同的有效日期。
        - **局限性**：历史相关性和协方差不保证未来稳定，优化权重对估计误差敏感。

        详细公式见 docs/formulas.md。压力测试将在下一阶段实现。
        """
    )


def _format_performance(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in ["annual_return", "annual_volatility", "max_drawdown"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    for column in ["sharpe", "sortino", "calmar"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    return formatted


def _target_return_input(minimum_return: float, maximum_return: float) -> float:
    if abs(maximum_return - minimum_return) <= 1e-10:
        st.info(f"当前约束下只有一个可行预期收益率：{minimum_return:.2%}。")
        return minimum_return
    minimum_percent = float(minimum_return * 100.0)
    maximum_percent = float(maximum_return * 100.0)
    default_percent = (minimum_percent + maximum_percent) / 2.0
    step = max((maximum_percent - minimum_percent) / 100.0, 0.01)
    selected = st.number_input(
        "目标年化预期收益率（%）",
        min_value=minimum_percent,
        max_value=maximum_percent,
        value=default_percent,
        step=step,
        format="%.2f",
    )
    return float(selected / 100.0)


def _render_frontier(frontier: EfficientFrontierResult) -> None:
    metrics = frontier.metrics_frame()
    st.subheader("有效前沿")
    st.plotly_chart(efficient_frontier_chart(metrics), width="stretch")
    display = metrics.copy()
    for column in ["target_return", "annual_return", "annual_volatility"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    st.dataframe(display, width="stretch")

    weights = frontier.weights_frame().add_prefix("weight_").reset_index()
    export = pd.concat([metrics, weights.drop(columns="target_return")], axis=1)
    st.download_button(
        "下载有效前沿 CSV",
        export.to_csv(index=False).encode("utf-8-sig"),
        file_name="efficient_frontier.csv",
        mime="text/csv",
    )
