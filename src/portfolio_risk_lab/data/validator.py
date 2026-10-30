"""价格表的领域校验。"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """输入数据不满足金融分析要求。"""


def validate_price_frame(prices: pd.DataFrame) -> None:
    """校验宽表价格；允许缺失值，但不允许无穷值和非正价格。"""
    if prices.empty:
        raise DataValidationError("价格数据为空，缺失值处理后没有可用交易日。")
    if prices.shape[1] == 0:
        raise DataValidationError("CSV 至少需要一个资产价格列。")
    if prices.columns.has_duplicates:
        raise DataValidationError("资产列名存在重复，请为每个资产使用唯一列名。")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataValidationError("价格表索引必须是日期。")
    if prices.index.has_duplicates:
        raise DataValidationError("日期存在重复；请先明确每个日期应保留哪条记录。")
    if not prices.index.is_monotonic_increasing:
        raise DataValidationError("日期必须按升序排列。")

    values = prices.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise DataValidationError("价格中包含正无穷或负无穷。")
    non_missing = values[~np.isnan(values)]
    if non_missing.size == 0:
        raise DataValidationError("所有资产价格均为空。")
    if (non_missing <= 0).any():
        raise DataValidationError("复权收盘价必须严格大于 0。")
