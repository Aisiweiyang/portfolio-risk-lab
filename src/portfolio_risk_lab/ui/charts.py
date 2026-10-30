"""集中构造 Plotly 图表，避免页面函数混入绘图细节。"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def cumulative_return_chart(cumulative: pd.DataFrame) -> go.Figure:
    long_frame = (
        cumulative.rename_axis("Date")
        .reset_index()
        .melt("Date", var_name="资产", value_name="累计收益")
    )
    figure = px.line(
        long_frame,
        x="Date",
        y="累计收益",
        color="资产",
        title="累计收益曲线",
    )
    figure.update_yaxes(tickformat=".1%")
    return figure


def correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        data=go.Heatmap(
            z=correlation.to_numpy(),
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=correlation.round(2).to_numpy(),
            texttemplate="%{text}",
        )
    )
    figure.update_layout(
        title="资产收益 Pearson 相关性",
        xaxis_title="资产",
        yaxis_title="资产",
    )
    return figure


def drawdown_chart(drawdowns: pd.Series, asset: str) -> go.Figure:
    figure = px.area(
        drawdowns.rename("回撤").rename_axis("Date").reset_index(),
        x="Date",
        y="回撤",
        title=f"{asset} 回撤曲线",
    )
    figure.update_yaxes(tickformat=".1%")
    return figure


def portfolio_weights_chart(weights: pd.DataFrame) -> go.Figure:
    long_frame = (
        weights.rename_axis("资产").reset_index().melt("资产", var_name="组合", value_name="权重")
    )
    figure = px.bar(
        long_frame,
        x="资产",
        y="权重",
        color="组合",
        barmode="group",
        title="组合权重比较",
    )
    figure.update_yaxes(tickformat=".0%")
    return figure


def efficient_frontier_chart(metrics: pd.DataFrame) -> go.Figure:
    figure = px.line(
        metrics,
        x="annual_volatility",
        y="annual_return",
        markers=True,
        custom_data=["target_return", "status", "solver"],
        title="有效前沿：预期收益与年化波动率",
        labels={
            "annual_volatility": "年化波动率",
            "annual_return": "年化预期收益",
        },
    )
    figure.update_traces(
        hovertemplate=(
            "年化波动率：%{x:.2%}<br>"
            "年化预期收益：%{y:.2%}<br>"
            "目标收益率：%{customdata[0]:.2%}<br>"
            "状态：%{customdata[1]}<br>"
            "求解器：%{customdata[2]}<extra></extra>"
        )
    )
    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")
    return figure


def backtest_net_value_chart(net_value: pd.DataFrame) -> go.Figure:
    long_frame = (
        net_value.rename_axis("Date").reset_index().melt("Date", var_name="组合", value_name="净值")
    )
    figure = px.line(
        long_frame,
        x="Date",
        y="净值",
        color="组合",
        title="样本外净值曲线（初始净值 1）",
    )
    figure.update_yaxes(tickformat=".3f")
    return figure


def backtest_drawdown_chart(drawdowns: pd.DataFrame) -> go.Figure:
    long_frame = (
        drawdowns.rename_axis("Date").reset_index().melt("Date", var_name="组合", value_name="回撤")
    )
    figure = px.line(
        long_frame,
        x="Date",
        y="回撤",
        color="组合",
        title="样本外回撤曲线",
    )
    figure.update_yaxes(tickformat=".1%")
    return figure


def backtest_turnover_chart(turnover: pd.Series) -> go.Figure:
    frame = turnover.rename("成交名义金额比例").rename_axis("Date").reset_index()
    figure = px.bar(
        frame,
        x="Date",
        y="成交名义金额比例",
        title="调仓成交名义金额比例",
    )
    figure.update_yaxes(tickformat=".1%")
    return figure


def rolling_weights_chart(weights: pd.DataFrame) -> go.Figure:
    long_frame = (
        weights.rename_axis("Date")
        .reset_index()
        .melt("Date", var_name="资产", value_name="期末权重")
    )
    figure = px.area(
        long_frame,
        x="Date",
        y="期末权重",
        color="资产",
        title="策略期末权重变化",
    )
    figure.update_yaxes(tickformat=".0%")
    return figure
