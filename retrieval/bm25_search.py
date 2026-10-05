import re
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from ingestion.chunker import DocumentChunk

class BM25SearchEngine:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [token for token in cleaned.split() if len(token) > 1]
        return tokens

    def index_chunks(self, chunks: List[DocumentChunk]):
        self.chunks = list(chunks)
        self.corpus_tokens = [self.tokenize(c.text) for c in self.chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0.0:
                results.append((self.chunks[idx], float(scores[idx])))
        return results
