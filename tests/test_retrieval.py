import pytest
from ingestion.chunker import DocumentChunk
from retrieval.bm25_search import BM25SearchEngine
from retrieval.vector_search import VectorSearchEngine
from retrieval.hybrid import HybridRetriever

@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            doc_id="d1",
            source="security_policy.txt",
            chunk_index=0,
            text="Multi-factor authentication MFA using hardware tokens or FIDO2 is mandatory for admin access.",
            metadata={"source": "security_policy.txt", "chunk_index": "0"}
        ),
        DocumentChunk(
            doc_id="d2",
            source="procurement.txt",
            chunk_index=0,
            text="Vendors must uphold 99.99% monthly availability SLAs or face financial invoice discounts.",
            metadata={"source": "procurement.txt", "chunk_index": "0"}
        ),
        DocumentChunk(
            doc_id="d3",
            source="finance.txt",
            chunk_index=0,
            text="Accounting and financial records must be archived in immutable WORM storage for 7 years.",
            metadata={"source": "finance.txt", "chunk_index": "0"}
        )
    ]

def test_bm25_search(sample_chunks):
    bm25 = BM25SearchEngine()
    bm25.index_chunks(sample_chunks)
    results = bm25.search("hardware MFA tokens", top_k=2)
    assert len(results) >= 1
    assert results[0][0].source == "security_policy.txt"

def test_vector_search(sample_chunks):
    vec = VectorSearchEngine()
    vec.index_chunks(sample_chunks)
    results = vec.search("cloud supplier uptime commitment", top_k=2)
    assert len(results) >= 1
    assert results[0][0].source == "procurement.txt"

def test_hybrid_rrf_fusion(sample_chunks):
    hybrid = HybridRetriever(rrf_k=60)
    hybrid.index_chunks(sample_chunks)
    results = hybrid.search("7 years immutable financial retention", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.source == "finance.txt"
    assert results[0].rrf_score > 0
    assert results[0].bm25_rank is not None or results[0].vector_rank is not None
