import pytest
from pathlib import Path
from analytics.data_engine import DataEngine
from analytics.query_generator import AnalyticsQueryGenerator
from core.config import settings

@pytest.fixture
def data_engine():
    engine = DataEngine()  # In-memory SQLite
    csv_path = settings.SAMPLES_DIR / "q3_financial_results.csv"
    engine.load_csv(csv_path, table_name="financials")
    yield engine
    engine.close()

def test_load_and_inspect_schema(data_engine):
    tables = data_engine.list_tables()
    assert "financials" in tables
    
    info = data_engine.get_table_info("financials")
    assert info is not None
    assert info.row_count == 12
    assert "revenue" in info.columns
    assert "region" in info.columns

def test_safe_select_execution(data_engine):
    res = data_engine.execute_query("SELECT region, revenue FROM financials WHERE region = 'North America'")
    assert res["row_count"] == 3
    assert "revenue" in res["columns"]

def test_sql_injection_guard(data_engine):
    with pytest.raises(ValueError, match="Security restriction"):
        data_engine.execute_query("DROP TABLE financials")

    with pytest.raises(ValueError, match="Security restriction"):
        data_engine.execute_query("DELETE FROM financials WHERE region = 'Europe'")

def test_analytics_query_generation(data_engine):
    gen = AnalyticsQueryGenerator(data_engine)
    res = gen.generate_and_execute("Total revenue by region")
    assert "data" in res
    assert len(res["data"]) > 0
    assert "sum_revenue" in res["data"][0]
