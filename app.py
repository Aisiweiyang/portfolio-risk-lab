"""Portfolio Risk Lab 的本地 Streamlit 入口。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st

from portfolio_risk_lab.analytics.returns import simple_returns
from portfolio_risk_lab.analytics.risk import complete_return_frame
from portfolio_risk_lab.config import AnalysisConfig
from portfolio_risk_lab.data.loader import (
    LoadedPriceData,
    MissingPolicy,
    load_price_csv,
)
from portfolio_risk_lab.data.validator import DataValidationError
from portfolio_risk_lab.ui.pages import (
    render_analytics,
    render_backtest,
    render_data_quality,
    render_help,
    render_portfolio,
)

st.set_page_config(page_title="投资组合优化与风险研究平台", page_icon="📊", layout="wide")

ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = ROOT / "sample_data" / "sample_prices.csv"


@st.cache_data(show_spinner=False)
def load_prices_cached(
    content: bytes,
    date_column: str,
    missing_policy: str,
    encoding: str,
) -> LoadedPriceData:
    """按文件内容和用户参数缓存确定性的 CSV 清洗结果。"""
    return load_price_csv(
        BytesIO(content),
        date_column=date_column,
        missing_policy=MissingPolicy(missing_policy),
        encoding=encoding,
    )


def main() -> None:
    st.title("投资组合优化与风险研究平台")
    st.caption("Portfolio Risk Lab · 本地研究版")
    st.warning("仅用于研究、教学和模拟，不构成投资建议；不连接券商，也不执行真实交易。")

    with st.sidebar:
        st.header("数据与参数")
        uploaded = st.file_uploader("上传价格 CSV", type=["csv"])
        date_column = st.text_input("日期列名", value="Date")
        encoding_label = st.selectbox("CSV 编码", ["UTF-8 / UTF-8 BOM", "GB18030"])
        encoding = {
            "UTF-8 / UTF-8 BOM": "utf-8-sig",
            "GB18030": "gb18030",
        }[encoding_label]
        policy_label = st.selectbox(
            "缺失值处理",
            ["删除任一资产缺失的日期", "保留缺失值", "前向填充后删除开头缺口"],
        )
        policy = {
            "删除任一资产缺失的日期": MissingPolicy.DROP_ROWS,
            "保留缺失值": MissingPolicy.KEEP,
            "前向填充后删除开头缺口": MissingPolicy.FORWARD_FILL,
        }[policy_label]
        st.caption("前向填充可能把停牌或跨市场休市解释为零收益，仅在明确需要时使用。")
        periods = st.number_input("每年交易周期", min_value=2, max_value=366, value=252)
        risk_free = st.number_input("年化无风险利率", value=0.02, step=0.005, format="%.3f")
        confidence = st.slider("VaR/CVaR 置信水平", 0.80, 0.99, 0.95, 0.01)

    config = AnalysisConfig(
        periods_per_year=int(periods),
        annual_risk_free_rate=float(risk_free),
        var_confidence_level=float(confidence),
    )
    content = uploaded.getvalue() if uploaded is not None else SAMPLE_FILE.read_bytes()
    source_name = uploaded.name if uploaded is not None else "合成演示数据 sample_prices.csv"

    try:
        loaded = load_prices_cached(content, date_column, policy.value, encoding)
        returns = simple_returns(loaded.prices)
        complete_returns = complete_return_frame(returns)
        if len(complete_returns) < 2:
            raise DataValidationError(
                "所有资产共同完整的收益观测少于 2 个，无法计算协方差和组合优化；"
                "请扩大数据区间或调整缺失值策略。"
            )
    except (DataValidationError, ValueError) as exc:
        st.error(f"数据无法用于分析：{exc}")
        st.stop()

    st.info(f"当前数据源：{source_name}")
    if uploaded is None:
        st.caption("内置数据为固定种子生成的虚构价格，仅演示功能，不代表真实行情或策略业绩。")
    overview_tab, analytics_tab, portfolio_tab, backtest_tab, help_tab = st.tabs(
        ["数据质量", "收益与风险", "组合优化", "滚动回测", "参数说明"]
    )
    with overview_tab:
        render_data_quality(loaded)
    with analytics_tab:
        render_analytics(loaded, returns, config)
    with portfolio_tab:
        render_portfolio(returns, config)
    with backtest_tab:
        render_backtest(complete_returns, config)
    with help_tab:
        render_help()


if __name__ == "__main__":
    main()
