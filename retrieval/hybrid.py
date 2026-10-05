from typing import List, Dict, Optional
from pydantic import BaseModel
from ingestion.chunker import DocumentChunk
from retrieval.bm25_search import BM25SearchEngine
from retrieval.vector_search import VectorSearchEngine

class HybridSearchResult(BaseModel):
    chunk: DocumentChunk
    rrf_score: float
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None

class HybridRetriever:
    def __init__(self, rrf_k: int = 60, vector_weight: float = 1.0, bm25_weight: float = 1.0):
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.bm25_engine = BM25SearchEngine()
        self.vector_engine = VectorSearchEngine()
        self.all_chunks: List[DocumentChunk] = []

    def index_chunks(self, chunks: List[DocumentChunk]):
        self.all_chunks = list(chunks)
        self.bm25_engine.index_chunks(chunks)
        self.vector_engine.index_chunks(chunks)

    def search(self, query: str, top_k: int = 5) -> List[HybridSearchResult]:
        if not self.all_chunks:
            return []

        # Candidate pool is 3x top_k to ensure high-recall fusion
        pool_size = max(top_k * 3, 10)
        bm25_results = self.bm25_engine.search(query, top_k=pool_size)
        vector_results = self.vector_engine.search(query, top_k=pool_size)

        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        bm25_ranks: Dict[str, int] = {}
        vector_ranks: Dict[str, int] = {}
        bm25_scores: Dict[str, float] = {}
        vector_scores: Dict[str, float] = {}

        # 1. Process BM25 rankings
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            bm25_ranks[cid] = rank
            bm25_scores[cid] = score
            rrf_contrib = self.bm25_weight / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib

        # 2. Process Vector rankings
        for rank, (chunk, score) in enumerate(vector_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            vector_ranks[cid] = rank
            vector_scores[cid] = score
            rrf_contrib = self.vector_weight / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib

        # 3. Sort by fused RRF score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        final_results = []
        for cid in sorted_cids[:top_k]:
            final_results.append(HybridSearchResult(
                chunk=chunk_map[cid],
                rrf_score=round(fused_scores[cid], 6),
                vector_rank=vector_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                vector_score=round(vector_scores[cid], 4) if cid in vector_scores else None,
                bm25_score=round(bm25_scores[cid], 4) if cid in bm25_scores else None
            ))

        return final_results
