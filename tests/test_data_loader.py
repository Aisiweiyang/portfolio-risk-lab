from io import BytesIO, StringIO

import pandas as pd
import pytest

from portfolio_risk_lab.data.loader import MissingPolicy, load_price_csv
from portfolio_risk_lab.data.validator import DataValidationError


def test_load_sorts_dates_and_drops_incomplete_rows() -> None:
    source = StringIO("Date,A,B\n2025-01-03,101,\n2025-01-02,100,50\n2025-01-06,102,51\n")

    loaded = load_price_csv(source, missing_policy=MissingPolicy.DROP_ROWS)

    assert list(loaded.prices.index) == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-06")]
    assert loaded.original_rows == 3
    assert loaded.output_rows == 2
    assert loaded.missing_before == 1
    assert loaded.missing_after == 0


def test_keep_missing_does_not_fill_price() -> None:
    source = StringIO("Date,A,B\n2025-01-02,100,50\n2025-01-03,101,\n")

    loaded = load_price_csv(source, missing_policy=MissingPolicy.KEEP)

    assert pd.isna(loaded.prices.loc[pd.Timestamp("2025-01-03"), "B"])
    assert loaded.missing_after == 1


def test_forward_fill_only_when_explicitly_selected() -> None:
    source = StringIO("Date,A,B\n2025-01-02,100,50\n2025-01-03,101,\n")

    loaded = load_price_csv(source, missing_policy=MissingPolicy.FORWARD_FILL)

    assert loaded.prices.loc[pd.Timestamp("2025-01-03"), "B"] == pytest.approx(50.0)


def test_duplicate_original_header_is_rejected_before_pandas_mangles_it() -> None:
    source = StringIO("Date,A,A\n2025-01-02,100,101\n")

    with pytest.raises(DataValidationError, match="原始表头存在重复列名"):
        load_price_csv(source)


def test_utf8_bom_and_gb18030_are_explicitly_supported() -> None:
    utf8_source = BytesIO("\ufeffDate,资产A\n2025-01-02,100\n".encode())
    gb_source = BytesIO("Date,资产A\n2025-01-02,100\n".encode("gb18030"))

    utf8_loaded = load_price_csv(utf8_source)
    gb_loaded = load_price_csv(gb_source, encoding="gb18030")

    assert list(utf8_loaded.prices.columns) == ["资产A"]
    assert list(gb_loaded.prices.columns) == ["资产A"]


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("", "CSV 文件为空"),
        ("When,A\n2025-01-02,100\n", "缺少日期列"),
        ("Date,A\n2025-01-02,-100\n", "必须严格大于 0"),
        ("Date,A\n2025-01-02,100\n2025-01-02,101\n", "日期存在重复"),
        ("Date,A\n2025-01-02,not-a-price\n", "包含非数值价格"),
    ],
)
def test_invalid_csv_has_clear_error(csv_text: str, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        load_price_csv(StringIO(csv_text))
