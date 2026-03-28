"""Tests for extract_holdings() with CSV and Excel inputs."""
import io
import pytest
import pandas as pd


def _make_csv_file(content, name="test.csv"):
    """Create a file-like object simulating an uploaded CSV."""
    f = io.BytesIO(content.encode("utf-8"))
    f.name = name
    f.seek(0)
    return f


def _make_excel_file(df, name="test.xlsx"):
    """Create a file-like object simulating an uploaded Excel file."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.name = name
    buf.seek(0)
    return buf


class TestCSVExtraction:
    def test_basic_extraction(self, extract_holdings):
        csv = "scheme_name,invested,current_value\nHDFC Flexi Cap,10000,12000\nSBI Large Cap,20000,22000"
        result = extract_holdings(_make_csv_file(csv))
        assert len(result) == 2
        assert result[0]["name"] == "HDFC Flexi Cap"

    def test_fund_column_detection(self, extract_holdings):
        csv = "fund_name,amount\nParag Parikh Flexi,50000"
        result = extract_holdings(_make_csv_file(csv))
        assert len(result) == 1
        assert result[0]["name"] == "Parag Parikh Flexi"

    def test_category_column(self, extract_holdings):
        csv = "scheme_name,category\nHDFC Flexi Cap,Flexi Cap"
        result = extract_holdings(_make_csv_file(csv))
        assert result[0].get("category") == "Flexi Cap"

    def test_nan_rows_skipped(self, extract_holdings):
        csv = "scheme_name,invested\nHDFC Flexi Cap,10000\n,\n,5000"
        result = extract_holdings(_make_csv_file(csv))
        assert len(result) == 1

    def test_comma_in_amount(self, extract_holdings):
        csv = 'scheme_name,invested\nHDFC Flexi Cap,"1,00,000"'
        result = extract_holdings(_make_csv_file(csv))
        assert result[0].get("invested") == 100000.0

    def test_invalid_amount_handled(self, extract_holdings):
        csv = "scheme_name,invested\nHDFC Flexi Cap,N/A"
        result = extract_holdings(_make_csv_file(csv))
        assert len(result) == 1
        assert "invested" not in result[0]

    def test_first_column_fallback(self, extract_holdings):
        csv = "name_of_holding,amount\nSome Fund,10000"
        result = extract_holdings(_make_csv_file(csv))
        assert len(result) == 1
        assert result[0]["name"] == "Some Fund"

    def test_column_whitespace_stripped(self, extract_holdings):
        csv = "  scheme_name  ,  invested  \nHDFC Flexi Cap,10000"
        result = extract_holdings(_make_csv_file(csv))
        assert len(result) == 1


class TestExcelExtraction:
    def test_basic_extraction(self, extract_holdings):
        df = pd.DataFrame({
            "scheme_name": ["HDFC Flexi Cap", "SBI Large Cap"],
            "invested": [10000, 20000],
        })
        result = extract_holdings(_make_excel_file(df))
        assert len(result) == 2

    def test_current_value_column(self, extract_holdings):
        df = pd.DataFrame({
            "scheme_name": ["HDFC Flexi Cap"],
            "current_value": [12000],
        })
        result = extract_holdings(_make_excel_file(df))
        assert result[0].get("current") == 12000.0


class TestUnsupportedFormat:
    def test_txt_file(self, extract_holdings):
        f = io.BytesIO(b"some text")
        f.name = "test.txt"
        result = extract_holdings(f)
        assert result == []
