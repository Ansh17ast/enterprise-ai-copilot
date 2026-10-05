import re
from enum import Enum
from typing import List
from pydantic import BaseModel

class QueryIntent(str, Enum):
    UNSTRUCTURED_RAG = "unstructured_rag"
    STRUCTURED_ANALYTICS = "structured_analytics"
    HYBRID_COMPOSITE = "hybrid_composite"

class RoutingDecision(BaseModel):
    intent: QueryIntent
    confidence: float
    reason: str
    target_tables: List[str] = []

class QueryRouter:
    ANALYTICS_KEYWORDS = [
        "total", "sum", "average", "avg", "mean", "count", "maximum", "max",
        "minimum", "min", "highest", "lowest", "revenue", "cost", "spend",
        "margin", "profit", "headcount", "budget", "trend", "breakdown",
        "by region", "by department", "by vendor", "by quarter", "table", "csv", "sql"
    ]

    DOCUMENT_KEYWORDS = [
        "policy", "procedure", "guideline", "protocol", "sla", "security",
        "retention", "compliance", "clause", "contract", "requirement",
        "definition", "explain", "describe", "summary", "why", "how", "what is the process"
    ]

    def route_query(self, query: str, available_tables: List[str] = None) -> RoutingDecision:
        q_lower = query.lower()
        analytics_matches = sum(1 for kw in self.ANALYTICS_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", q_lower))
        doc_matches = sum(1 for kw in self.DOCUMENT_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", q_lower))

        matched_tables = []
        if available_tables:
            for t in available_tables:
                if re.search(rf"\b{re.escape(t.lower())}\b", q_lower):
                    matched_tables.append(t)

        # Decision Logic
        if analytics_matches > 0 and doc_matches > 0:
            return RoutingDecision(
                intent=QueryIntent.HYBRID_COMPOSITE,
                confidence=0.88,
                reason="Query contains both quantitative analytical metrics and qualitative document/policy keywords.",
                target_tables=matched_tables
            )
        elif analytics_matches >= 2 or (analytics_matches >= 1 and matched_tables):
            return RoutingDecision(
                intent=QueryIntent.STRUCTURED_ANALYTICS,
                confidence=0.92,
                reason="Query indicates quantitative aggregations, calculations, or structured tabular analysis.",
                target_tables=matched_tables
            )
        else:
            return RoutingDecision(
                intent=QueryIntent.UNSTRUCTURED_RAG,
                confidence=0.85,
                reason="Query seeks textual information, procedural explanation, or contextual policy retrieval.",
                target_tables=matched_tables
            )
