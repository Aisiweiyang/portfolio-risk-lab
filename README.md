# Portfolio Risk Lab · 投资组合优化与风险研究平台

[![Checks](https://github.com/Aisiweiyang/portfolio-risk-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Aisiweiyang/portfolio-risk-lab/actions/workflows/ci.yml)

面向金融工程学习和本地研究的 Python / Streamlit 应用：导入价格 CSV，检查收益与风险，求解带权重约束的组合，并通过可审计时序的滚动回测观察成本与权重漂移。

A local research and teaching app for portfolio risk analysis, constrained optimization, and rolling backtests. Chinese UI; reproducible synthetic demo data; Python 3.11–3.12; MIT licensed.

> 当前版本 **0.2.0，早期研究版**。仅用于研究、教学和模拟，不构成投资建议；不连接券商或执行真实交易。内置数据为合成价格，演示图表和回测数字不代表真实市场业绩。

## 快速开始

下载本仓库源码或源码发布包，解压并进入包含 `app.py` 的目录。需要 Python **3.11 或 3.12** 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
uv sync --locked --python 3.12
uv run --locked streamlit run app.py
```

浏览器访问 [http://127.0.0.1:8501](http://127.0.0.1:8501)。Windows 安装依赖后，也可双击 `运行.cmd`；可传入 `--server.port 8511` 等 Streamlit 参数。

没有 uv 时，可用标准 CPython 和 pip 安装源码：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m streamlit run app.py
```

```bash
# macOS / Linux
.venv/bin/python -m pip install -e .
.venv/bin/python -m streamlit run app.py
```

uv 安装使用提交的 `uv.lock` 复现依赖；pip 安装按 `pyproject.toml` 的版本范围解析，结果可能不同。不要复制其他电脑的 `.venv` 或 `.python`。本仓库不包含解释器、虚拟环境和下载缓存。

## 体验流程

1. 直接启动，查看默认的三资产**合成演示数据**及数据质量表。
2. 在“收益与风险”查看绩效、相关矩阵、回撤及历史 VaR/CVaR。
3. 在“组合优化”选择最小方差、目标收益率或有效前沿，点击运行并查看约束残差。
4. 在“滚动回测”设置训练窗口、再平衡频率和成本，查看净值、漂移权重及时序审计。
5. 用自己的合规授权数据替换示例；结果可通过页面下载为 CSV。

## 已实现

| 模块 | 能力 |
| --- | --- |
| 数据 | 价格宽表 CSV；日期排序；重复日期、原始重复表头、非法价格及空文件校验；UTF-8/BOM 和 GB18030 |
| 缺失值 | 完整行、保留缺失、显式前向填充三种策略；收益不隐式填充 |
| 分析 | 简单/对数/累计收益，几何年化收益与波动率，Sharpe、Sortino、Calmar、最大回撤，历史 VaR/CVaR |
| 组合 | 共同样本收益向量与协方差，相关矩阵，风险贡献，等权及受约束最小方差、目标收益率、有效前沿 |
| 求解诊断 | CLARABEL 优先、SCS 备选；求解状态、权重和、上下界及目标收益残差；不静默放松约束 |
| 回测 | 等权与最小方差滚动策略，下一期生效、自然权重漂移、比例交易成本、等权买入持有基准及时序审计 |
| 界面 | 中文页面、交互图表、按内容缓存、点击后求解和 CSV 下载 |

## 数据格式与来源

输入是日期列加若干价格列的宽表。真实数据建议使用口径一致的复权收盘价；程序只校验数值，无法判断复权方式或数据授权。

```csv
Date,ETF_A,ETF_B,INDEX_C
2025-01-02,100.00,80.00,120.00
2025-01-03,100.70,79.80,120.60
```

日期必须可解析且唯一，至少包含一个资产；非缺失价格必须为有限正数。原始表头不得为空、重复或含首尾空格。程序不猜测重复日期应该保留哪一行。

- **完整行**：删除任一资产缺失的日期。
- **保留缺失**：单资产指标使用各自有效观测，组合估计仍使用共同完整收益日。
- **前向填充**：仅显式选择时使用；可能把停牌或跨市场休市解释为零收益。

内置 [sample_prices.csv](sample_data/sample_prices.csv) 由仓库脚本生成，包含三个虚构资产、261 个价格观测，来源和参数见 [示例数据说明](sample_data/README.md)。不会自动下载行情。真实输入、持仓和导出文件请放入已忽略的 `data/`、`uploads/`、`exports/` 或 `results/` 目录。

## 研究口径与限制

完整定义见 [公式、假设与局限](docs/formulas.md)。影响解释的约定：

- 每年默认 252 期，无风险利率为有效年利率；频率与日历应由使用者核对。
- Sharpe/Sortino 分子采用几何年化收益减年化无风险利率，与部分工具的算术定义不同。
- 优化收益向量采用共同样本上的**逐资产几何年化收益**。其权重加权值是收益代理，不等于组合实际几何年化收益，也不是未来收益预测。
- VaR/CVaR 按单周期损失计算：正数为损失，负数为盈利。历史样本不能覆盖所有未来极端情况。
- 权重只使用截至决策日的训练收益，从下一期开始生效；测试包含修改未来数据后历史权重不变的检查。输入数据本身的前视、幸存者偏差和调参偏差仍需使用者控制。
- 交易成本按买卖分别计入的成交名义金额收取；未模拟市场冲击、成交限制、税费或停牌，不能据此推断真实可成交表现。

## 开发与检查

```bash
uv sync --locked --extra dev --python 3.12
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/generate_sample_data.py --check
uv build
uv run --locked python scripts/verify_distributions.py
```

测试覆盖数据校验、指标与风险、组合约束、有效前沿、回测时序/成本/漂移，以及 Streamlit 默认页面和主要计算交互。[GitHub Actions](.github/workflows/ci.yml) 配置了 Windows/Linux × Python 3.11/3.12 的检查；配置存在不代表各平台已运行通过，请以实际检查结果为准。

应用从源码目录运行。构建产生的 `.tar.gz` 源码包包含 `app.py`、示例、文档和测试；`.whl` 仅提供 `portfolio_risk_lab` 计算包，不含完整应用入口。发布步骤见 [发布指南](docs/releasing.md)。

## 常见问题

- **约束不可行**：`资产数 × 最大权重 < 1` 时直接报错；目标收益超出可行区间时不会自动修正。
- **共同收益观测不足**：协方差至少需要两期共同完整收益，回测还需在训练窗口后保留至少两期。
- **零波动或无下行风险**：部分比率没有定义，结果可能为空值或明确错误。
- **首次求解慢**：CVXPY/SciPy 在运行计算时加载；首次加载通常较慢。
- **`optimal_inaccurate`**：查看求解诊断；超过约束容差的结果会报错。
- **启动失败或搬迁项目后报错**：进入正确目录重新执行 `uv sync --locked --python 3.12`，使用新建的环境；确保解释器是标准 CPython。
- **端口被占用**：运行 `uv run --locked streamlit run app.py --server.port 8511`。

## 项目结构

```text
portfolio-risk-lab/
├── app.py                       # Streamlit 入口
├── src/portfolio_risk_lab/
│   ├── data/                    # 导入、校验、缺失值
│   ├── analytics/               # 收益、绩效、风险
│   ├── optimization/            # 约束与求解
│   ├── backtest/                # 时序、漂移、成本
│   ├── services/                # 共同样本与计算编排
│   └── ui/                      # 页面和图表
├── tests/
├── sample_data/                 # 合成示例及来源说明
├── scripts/generate_sample_data.py
├── docs/                        # 公式与发布指南
├── .github/                     # CI、Issue 与 PR 模板
├── pyproject.toml
├── uv.lock
└── LICENSE
```

## 后续方向

尚未实现：协方差收缩、指数加权估计、压力情景和本地实验记录。实盘交易、券商 API、预测、账号系统与公网部署不在当前范围内。

## 贡献与许可

参与方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，版本变化见 [CHANGELOG.md](CHANGELOG.md)。项目代码与生成的合成数据采用 [MIT License](LICENSE)（[许可证原文](https://opensource.org/license/mit)）；第三方依赖保留各自许可，用户导入数据的使用和再分发权限由其来源决定。
