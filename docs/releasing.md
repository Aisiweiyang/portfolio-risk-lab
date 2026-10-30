# 本地检查与发布

本指南准备可检查的发布材料。构建命令只在本地生成文件，不会创建远端仓库、上传源码或发布到 PyPI。

## 1. 验证候选版本

```bash
uv sync --locked --extra dev --python 3.12
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked python scripts/generate_sample_data.py --check
```

手动启动 `uv run --locked streamlit run app.py`，检查默认示例、组合优化、有效前沿、滚动回测及下载入口。内置数据应明确标为合成数据。

检查 `pyproject.toml` 与 `src/portfolio_risk_lab/__init__.py` 的版本一致，更新 `CHANGELOG.md`，确认 README 的能力与实际实现一致。跨平台结果以相应 CI 执行为准。

## 2. 构建与核对

```bash
uv build
uv run --locked python scripts/verify_distributions.py
```

输出位于 `dist/`：

- `portfolio_risk_lab-<版本>.tar.gz`：完整源码发布包；解压后可按 README 安装和启动应用。
- `portfolio_risk_lab-<版本>-py3-none-any.whl`：可安装计算库；不包含根目录的 Streamlit 应用与示例。

sdist 使用 `pyproject.toml` 中的路径白名单。检查解压内容包含许可证、入口、源码、示例、锁文件、文档与测试，并确认不含 `.venv`、`.python`、缓存、日志、凭据或私有输入。建议在新目录解压源码包，再运行一次安装与核心检查；不要直接打包整个工作目录。

Windows PowerShell 可生成分发文件摘要：

```powershell
Get-FileHash -Algorithm SHA256 dist\*.tar.gz,dist\*.whl
```

## 3. 公开仓库前检查

真实输入、持仓和导出结果放在忽略目录；`.gitignore` 不会移除已经跟踪的文件，发布时还应审阅实际待提交列表。MIT 只授予项目有权提供的内容，不能替代第三方依赖或数据的授权。

创建 GitHub 仓库时，建议名称 `portfolio-risk-lab`，描述：

> Local portfolio risk analysis, constrained optimization and rolling backtests. Python / Streamlit, Chinese UI, synthetic demo data.

建议主题：`portfolio-optimization`、`risk-management`、`backtesting`、`streamlit`、`financial-engineering`、`education`。

## 4. 远端发布

获得明确发布授权并确定账号和仓库后，再初始化/关联 Git、审阅首个提交和推送。首推后检查 `.github/workflows/ci.yml` 对应的四个任务；通过后可创建 `v<版本>` 标签和 GitHub Release，发布说明取自 `CHANGELOG.md`。

可附源码包及 SHA256 摘要。是否发布计算库 wheel 或到 PyPI 应单独决定，不默认将本项目注册为公开包。

## 参考

- [uv：锁文件与同步](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Hatch：构建配置与文件选择](https://hatch.pypa.io/latest/config/build/)
- [GitHub：checkout](https://github.com/actions/checkout) 与 [setup-python](https://github.com/actions/setup-python)
