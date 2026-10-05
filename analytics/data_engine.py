import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel

class TableInfo(BaseModel):
    table_name: str
    row_count: int
    columns: List[str]
    sample_rows: List[Dict[str, Any]]

class DataEngine:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path) if db_path else ":memory:"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def load_csv(self, file_path: Path, table_name: Optional[str] = None) -> str:
        name = table_name or file_path.stem.lower()
        # Clean table name to alphanumeric + underscore
        cleaned_name = re.sub(r"[^\w]", "_", name)
        
        df = pd.read_csv(file_path)
        # Clean column names
        df.columns = [re.sub(r"[^\w]", "_", c.strip().lower()) for c in df.columns]
        
        df.to_sql(cleaned_name, self.conn, if_exists="replace", index=False)
        return cleaned_name

    def list_tables(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        return [row[0] for row in cursor.fetchall()]

    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        cursor = self.conn.cursor()
        # Verify table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            return None

        # Column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = [row[1] for row in cursor.fetchall()]

        # Row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        # Sample rows
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample_rows = [dict(row) for row in cursor.fetchall()]

        return TableInfo(
            table_name=table_name,
            row_count=row_count,
            columns=cols,
            sample_rows=sample_rows
        )

    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        # SQL Injection & Mutation Safety Guard
        normalized = sql_query.strip().lower()
        disallowed_keywords = ["drop", "delete", "truncate", "update", "insert", "alter", "create", "grant", "revoke"]
        for kw in disallowed_keywords:
            # Check for standalone keyword
            if re.search(rf"\b{kw}\b", normalized):
                raise ValueError(f"Security restriction: {kw.upper()} operations are not permitted in read-only analytics mode.")

        if not normalized.startswith("select") and not normalized.startswith("with"):
            raise ValueError("Only SELECT or WITH query operations are permitted.")

        df = pd.read_sql_query(sql_query, self.conn)
        return {
            "query": sql_query,
            "columns": list(df.columns),
            "row_count": len(df),
            "data": df.to_dict(orient="records")
        }

    def close(self):
        if self.conn:
            self.conn.close()
