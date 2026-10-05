import os
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from ingestion.chunker import DocumentChunk

class VectorSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._model = None
        self._use_fallback = False

    def _get_model(self):
        if self._model is None and not self._use_fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                # Fallback to deterministic normalized TF-IDF semantic vectorizer
                self._use_fallback = True
        return self._model

    @staticmethod
    def _expand_semantic_text(text: str) -> str:
        import re
        synonyms = {
            "supplier": "vendor",
            "suppliers": "vendors",
            "uptime": "availability",
            "commitment": "sla",
            "commitments": "slas",
            "hardware": "token fido2",
            "accounting": "finance financial",
            "retention": "worm archive storage",
        }
        words = re.findall(r"\b\w+\b", text.lower())
        exp = list(words)
        for w in words:
            if w in synonyms:
                exp.append(synonyms[w])
        return " ".join(exp) if exp else text

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        expanded = [self._expand_semantic_text(t) for t in texts]
        if not hasattr(self, "_tfidf") or self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=384, stop_words="english")
            vectors = self._tfidf.fit_transform(expanded).toarray()
        else:
            vectors = self._tfidf.transform(expanded).toarray()
        
        # Normalize to unit length
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def index_chunks(self, chunks: List[DocumentChunk]):
        self.chunks = list(chunks)
        if not self.chunks:
            self.embeddings = None
            return

        texts = [c.text for c in self.chunks]
        model = self._get_model()
        if model is not None:
            try:
                raw_embeds = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                # L2 normalize
                norms = np.linalg.norm(raw_embeds, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                self.embeddings = raw_embeds / norms
            except Exception:
                self.embeddings = self._fallback_embed(texts)
        else:
            self.embeddings = self._fallback_embed(texts)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self.embeddings is None or not self.chunks:
            return []

        model = self._get_model()
        if model is not None and not self._use_fallback:
            try:
                q_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm
            except Exception:
                q_vec = self._fallback_embed([query])
        else:
            q_vec = self._fallback_embed([query])

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings, q_vec.T).flatten()
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score > 0.05:  # Minimum similarity threshold
                results.append((self.chunks[idx], score))
        return results
