# 合成演示数据

`sample_prices.csv` 由本仓库的 `scripts/generate_sample_data.py` 生成，随项目以 MIT 许可证分发。

- 三个虚构资产：`ETF_A`、`ETF_B`、`INDEX_C`，不对应真实证券。
- 固定随机种子 `20261030`；使用 Python 标准库，生成 261 个价格观测和 260 期收益。
- 日期从 2025-01-02 起，仅跳过周末；不模拟交易所节假日。
- 用共同随机冲击与各资产独立冲击构造对数收益，再复利生成严格为正的价格，保留四位小数。
- 不连接行情服务，不使用券商数据、真实账户或真实持仓。

重建及一致性检查：

```bash
python scripts/generate_sample_data.py
python scripts/generate_sample_data.py --check
```

这些数据只用于界面演示、安装验收和测试。图表、最优权重及样本外回测数字不构成真实市场结果；固定种子也不代表统计或经济显著性。
