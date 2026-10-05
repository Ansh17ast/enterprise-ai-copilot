import re
from typing import Optional, Dict, Any
from analytics.data_engine import DataEngine

class AnalyticsQueryGenerator:
    def __init__(self, data_engine: DataEngine):
        self.engine = data_engine

    def generate_and_execute(self, user_query: str, target_table: Optional[str] = None) -> Dict[str, Any]:
        tables = self.engine.list_tables()
        if not tables:
            return {
                "error": "No structured data tables available in the database.",
                "query": "",
                "data": []
            }

        # Resolve target table
        selected_table = None
        if target_table and target_table in tables:
            selected_table = target_table
        else:
            # Check table name mention in query
            for t in tables:
                if t in user_query.lower():
                    selected_table = t
                    break
            if not selected_table:
                selected_table = tables[0]  # Default to first available table

        table_info = self.engine.get_table_info(selected_table)
        columns = table_info.columns if table_info else []
        q_lower = user_query.lower()

        # Identify numeric column candidates
        numeric_cols = [c for c in columns if any(k in c for k in ["revenue", "cost", "spend", "margin", "profit", "amount", "headcount", "score", "rate", "count", "budget"])]
        category_cols = [c for c in columns if any(k in c for k in ["region", "department", "category", "vendor", "name", "quarter", "tier", "status", "division"])]

        # Intent detection
        sql = ""
        agg_func = "SUM"
        if "average" in q_lower or "avg" in q_lower or "mean" in q_lower:
            agg_func = "AVG"
        elif "maximum" in q_lower or "max" in q_lower or "highest" in q_lower:
            agg_func = "MAX"
        elif "minimum" in q_lower or "min" in q_lower or "lowest" in q_lower:
            agg_func = "MIN"
        elif "count" in q_lower or "number of" in q_lower:
            agg_func = "COUNT"

        target_num_col = None
        for nc in numeric_cols:
            if nc.replace("_", " ") in q_lower or nc in q_lower:
                target_num_col = nc
                break
        if not target_num_col and numeric_cols:
            target_num_col = numeric_cols[0]

        target_cat_col = None
        for cc in category_cols:
            if cc.replace("_", " ") in q_lower or cc in q_lower or f"by {cc}" in q_lower:
                target_cat_col = cc
                break
        if not target_cat_col and "by" in q_lower:
            # Try to extract the word after "by"
            match = re.search(r"\bby\s+([a-zA-Z_]+)", q_lower)
            if match and match.group(1) in columns:
                target_cat_col = match.group(1)

        # Build SQL Query
        if target_cat_col and target_num_col:
            sql = f"SELECT {target_cat_col}, ROUND({agg_func}({target_num_col}), 2) AS {agg_func.lower()}_{target_num_col} FROM {selected_table} GROUP BY {target_cat_col} ORDER BY {agg_func.lower()}_{target_num_col} DESC LIMIT 10"
        elif target_num_col:
            sql = f"SELECT ROUND({agg_func}({target_num_col}), 2) AS {agg_func.lower()}_{target_num_col} FROM {selected_table}"
        else:
            sql = f"SELECT * FROM {selected_table} LIMIT 10"

        try:
            result = self.engine.execute_query(sql)
            result["table_used"] = selected_table
            result["intent"] = agg_func
            return result
        except Exception as e:
            return {
                "error": str(e),
                "query": sql,
                "data": []
            }
