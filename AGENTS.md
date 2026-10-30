# Portfolio Risk Lab 协作约定

## 项目边界

- 仅用于金融工程研究、教学和模拟，不构成投资建议。
- 不连接券商、不执行真实交易、不保存密钥；仅在用户明确授权时上传源码或部署。
- 所有实现和生成数据仅写入本项目目录。

## 结构

- `app.py`：Streamlit 入口，只负责交互与结果展示。
- `src/portfolio_risk_lab/data/`：CSV 读取、校验和缺失值策略。
- `src/portfolio_risk_lab/analytics/`：收益、绩效、回撤和风险公式。
- `src/portfolio_risk_lab/backtest/`：滚动时序、权重漂移、交易成本和基准。
- `src/portfolio_risk_lab/optimization/`：组合约束与凸优化。
- `src/portfolio_risk_lab/services/`：统一样本与分析流程编排。
- `src/portfolio_risk_lab/ui/`：Streamlit 页面和 Plotly 图表。
- `tests/`：固定数据、无随机漂移的单元测试。
- `docs/`：公式、假设和局限性。

## Windows PowerShell 命令

```powershell
uv sync --locked --extra dev --python 3.12
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/generate_sample_data.py --check
uv run --locked streamlit run app.py
```

## 完成标准

改动必须职责清晰；组合预期收益与协方差必须使用同一估计样本；金融核心逻辑有公式说明和测试；输入错误可理解；不静默改变用户约束或数据策略；只有实际执行过的测试才能报告为通过。
