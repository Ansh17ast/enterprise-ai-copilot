import pytest
from routing.router import QueryRouter, QueryIntent

def test_router_unstructured():
    router = QueryRouter()
    res = router.route_query("What is our corporate data retention policy for financial logs?")
    assert res.intent == QueryIntent.UNSTRUCTURED_RAG

def test_router_structured():
    router = QueryRouter()
    res = router.route_query("What is the total revenue by region in Q3?")
    assert res.intent == QueryIntent.STRUCTURED_ANALYTICS

def test_router_hybrid():
    router = QueryRouter()
    res = router.route_query("Show total spend along with the vendor termination policy protocol")
    assert res.intent == QueryIntent.HYBRID_COMPOSITE
