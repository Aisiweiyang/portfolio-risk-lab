"""交易成本口径。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransactionCost:
    """一次调仓的成交名义金额比例和成本比例。"""

    traded_notional_ratio: float
    cost_fraction: float


def proportional_transaction_cost(
    previous_weights: pd.Series,
    target_weights: pd.Series,
    rate: float,
) -> TransactionCost:
    """按买卖名义金额计算固定比例成本。

    traded_notional_ratio = sum(abs(target_weight - previous_weight))
    cost_fraction = rate * traded_notional_ratio

    首次建仓的 previous_weights 为全零，因此全额建仓的成交名义金额比例为 1。
    后续调仓同时把卖出和买入分别计入成交名义金额。
    """
    if not np.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError("交易成本率必须是 [0, 1) 内的有限数值。")
    if (
        previous_weights.empty
        or target_weights.empty
        or previous_weights.index.has_duplicates
        or target_weights.index.has_duplicates
    ):
        raise ValueError("交易成本计算需要非空且资产标签不重复的权重。")
    if set(previous_weights.index) != set(target_weights.index):
        raise ValueError("调仓前权重与目标权重的资产集合必须一致。")
    previous = previous_weights.reindex(target_weights.index).to_numpy(dtype=float)
    target = target_weights.to_numpy(dtype=float)
    if not np.isfinite(previous).all() or not np.isfinite(target).all():
        raise ValueError("交易成本计算中的权重必须是有限数值。")

    traded_notional = float(np.abs(target - previous).sum())
    return TransactionCost(
        traded_notional_ratio=traded_notional,
        cost_fraction=float(rate * traded_notional),
    )
