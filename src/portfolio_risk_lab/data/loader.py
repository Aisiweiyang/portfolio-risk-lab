"""从 CSV 构造经过显式缺失值策略处理的价格宽表。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from io import StringIO
from pathlib import Path
from typing import IO

import pandas as pd

from portfolio_risk_lab.data.validator import DataValidationError, validate_price_frame


class MissingPolicy(StrEnum):
    """缺失价格处理策略。"""

    DROP_ROWS = "drop_rows"
    KEEP = "keep"
    FORWARD_FILL = "forward_fill"


@dataclass(frozen=True)
class LoadedPriceData:
    prices: pd.DataFrame
    original_rows: int
    output_rows: int
    missing_before: int
    missing_after: int
    policy: MissingPolicy


CsvSource = str | Path | IO[str] | IO[bytes]


def load_price_csv(
    source: CsvSource,
    *,
    date_column: str = "Date",
    missing_policy: MissingPolicy | str = MissingPolicy.DROP_ROWS,
    encoding: str = "utf-8-sig",
) -> LoadedPriceData:
    """读取“日期 + 一个或多个资产价格列”的宽表 CSV。"""
    text = _read_csv_text(source, encoding)
    _validate_original_header(text)
    try:
        raw = pd.read_csv(StringIO(text))
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("CSV 文件为空。") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"CSV 无法读取：{exc}") from exc

    if raw.empty:
        raise DataValidationError("CSV 没有数据行。")
    if raw.columns.has_duplicates:
        raise DataValidationError("CSV 存在重复列名。")
    if date_column not in raw.columns:
        raise DataValidationError(f"缺少日期列“{date_column}”。")
    if len(raw.columns) < 2:
        raise DataValidationError("CSV 至少需要日期列和一个资产价格列。")

    parsed_dates = pd.to_datetime(raw[date_column], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates:
        raise DataValidationError(f"日期列有 {invalid_dates} 个无法解析的值。")
    if parsed_dates.duplicated().any():
        duplicates = parsed_dates[parsed_dates.duplicated(keep=False)].dt.strftime("%Y-%m-%d")
        preview = ", ".join(duplicates.unique()[:3])
        raise DataValidationError(f"日期存在重复（例如 {preview}），未自动删除。")

    asset_columns = [column for column in raw.columns if column != date_column]
    prices = raw[asset_columns].copy()
    for column in asset_columns:
        converted = pd.to_numeric(prices[column], errors="coerce")
        introduced = converted.isna() & prices[column].notna()
        if introduced.any():
            raise DataValidationError(f"资产“{column}”包含非数值价格。")
        prices[column] = converted

    prices.index = pd.DatetimeIndex(parsed_dates, name=date_column)
    prices = prices.sort_index()
    validate_price_frame(prices)

    policy = MissingPolicy(missing_policy)
    original_rows = len(prices)
    missing_before = int(prices.isna().sum().sum())
    if policy is MissingPolicy.DROP_ROWS:
        prices = prices.dropna(axis=0, how="any")
    elif policy is MissingPolicy.FORWARD_FILL:
        prices = prices.ffill().dropna(axis=0, how="any")
    # KEEP 明确保留缺失值；后续计算不会隐式前向填充。

    validate_price_frame(prices)
    return LoadedPriceData(
        prices=prices,
        original_rows=original_rows,
        output_rows=len(prices),
        missing_before=missing_before,
        missing_after=int(prices.isna().sum().sum()),
        policy=policy,
    )


def _read_csv_text(source: CsvSource, encoding: str) -> str:
    try:
        if isinstance(source, (str, Path)):
            payload: str | bytes = Path(source).read_bytes()
        elif hasattr(source, "getvalue"):
            payload = source.getvalue()
        else:
            original_position = source.tell() if source.seekable() else None
            payload = source.read()
            if original_position is not None:
                source.seek(original_position)
    except OSError as exc:
        raise DataValidationError(f"CSV 无法读取：{exc}") from exc

    if isinstance(payload, str):
        return payload
    if not isinstance(payload, bytes):
        raise DataValidationError("CSV 输入必须是文本或字节流。")
    try:
        return payload.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"CSV 无法按 {encoding} 解码；请在界面选择正确编码。") from exc


def _validate_original_header(text: str) -> None:
    try:
        header = next(csv.reader(StringIO(text)))
    except (StopIteration, csv.Error) as exc:
        raise DataValidationError("CSV 文件为空或表头无法解析。") from exc

    normalized = [column.strip() for column in header]
    if any(not column for column in normalized):
        raise DataValidationError("CSV 表头包含空列名。")
    if normalized != header:
        raise DataValidationError("CSV 列名包含首尾空格，请先清理表头。")
    duplicates = sorted({column for column in header if header.count(column) > 1})
    if duplicates:
        preview = "、".join(duplicates[:3])
        raise DataValidationError(f"CSV 原始表头存在重复列名：{preview}。")
