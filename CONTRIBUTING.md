# 参与贡献

欢迎提交可复现的错误报告、公式勘误和范围明确的改进。讨论和代码注释可用中文或英文。

## 开发环境

使用 Python 3.11 或 3.12；推荐通过 [uv](https://docs.astral.sh/uv/) 和提交的锁文件安装：

```bash
uv sync --locked --extra dev --python 3.12
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/generate_sample_data.py --check
uv build
uv run --locked python scripts/verify_distributions.py
```

## 修改与验证

- 保持入口、展示、计算、数据校验和回测引擎的职责边界。
- 金融逻辑变动应同时更新 `docs/formulas.md`，并用可手算或固定输入的测试验证。
- 组合收益向量与协方差须使用同一组共同完整收益观测。
- 回测训练数据截至决策日，权重从下一期生效；不得根据未来收益修改历史权重。
- 不静默填充缺失数据、放松权重限制、修改目标收益率或跳过求解失败。
- 页面变动应验证默认示例及受影响的交互；新增功能应有真实可用实现。
- 报告实际执行的检查，明确指出未验证的平台或场景。

## Issue 和 Pull Request

错误报告请包含复现步骤、预期行为、实际行为、Python/操作系统版本和精简输入。优先使用合成数据；请勿提交账户、真实持仓、付费行情、个人路径、凭据或原始日志中的敏感信息。

Pull Request 请说明问题、改动后的行为和验证结果。涉及指标口径时说明公式和影响，不把演示数据或一次回测写成市场有效性证明。

提交贡献即表示你有权以本项目的 MIT 许可证提供该贡献。依赖保留各自的许可证；第三方数据需单独核实再分发权限。
