# System Architecture: Enterprise AI Analytics & Document Intelligence Copilot

## 1. Architectural Philosophy
Enterprise AI Copilot is designed to solve a fundamental deficiency in standard conversational RAG pipelines: **inability to accurately answer structured quantitative business questions and lack of verifiable, citation-backed refusal mechanisms.**

Standard vector search excels at vague semantic similarity but completely collapses on aggregation queries (e.g., *"What is total cloud spend in Q3?"* or *"Average margin by region"*). Conversely, traditional Business Intelligence (BI) dashboards lack the ability to synthesize unstructured compliance policies, SLA penalties, and legal contracts.

This system unifies both paradigms via an **Intent-Aware Hybrid Router**:

```
                              User Query
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Query Router   │
                         │ (Heuristic Regex│
                         │  + Intent ML)   │
                         └────────┬────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼ (UNSTRUCTURED_RAG)     ▼ (STRUCTURED_ANALYTICS) ▼ (HYBRID_COMPOSITE)
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Hybrid Search   │     │  SQL / Pandas    │     │  Dual Pipeline   │
│  - BM25 Lexical  │     │  - Schema Guard  │     │  - SQL Execution │
│  - Dense Vector  │     │  - Aggregations  │     │  - RAG Context   │
│  - RRF Fusion    │     │  - Safe SELECT   │     │  - Co-Synthesis  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Grounded QA Synthesizer │
                     │ - Strict Refusal Guard  │
                     │ - Explicit Citations    │
                     │ - No Hallucination      │
                     └────────────┬────────────┘
                                  │
                                  ▼
                         Verified Response
                     [Answer + Direct Sources]
```

---

## 2. Component Breakdown

### A. Ingestion & Document Processing (`ingestion/`)
- **Multi-Format Extraction**: Ingests `.pdf` (via PyMuPDF), `.csv` (via Pandas), `.txt`, and `.md`.
- **Boundary-Aware Sliding Window**: Breaks long passages into configurable chunks (default 500 characters, 100 overlap) respecting natural paragraph and sentence boundaries.
- **Context Metadata Propagation**: Preserves `doc_id`, `source`, `doc_type`, `char_start`, `char_end`, and `chunk_index` across every chunk for 100% auditable citations.

### B. Hybrid Retrieval Layer (`retrieval/`)
- **Lexical Search (BM25)**: Implements Okapi BM25 (`rank_bm25`) with token normalization and punctuation stripping to capture exact codes, numbers, and technical identifiers (e.g., `SOX`, `MFA`, `99.99%`).
- **Dense Vector Search**: Sentence embeddings (`all-MiniLM-L6-v2` / Cosine similarity) capturing semantic synonyms and conceptual queries.
- **Reciprocal Rank Fusion (RRF)**:
  $$\text{Score}_{\text{RRF}}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
  with constant $k = 60$. Ensures neither lexical frequency nor vector magnitude dominates the rank ordering.

### C. Structured Analytics Engine (`analytics/`)
- **In-Memory / Persistent SQLite**: Automates schema generation and CSV ingestion.
- **Mutation Safety Guard**: Rejects `DROP`, `DELETE`, `UPDATE`, `INSERT`, and `ALTER` statements to enforce read-only analytical safety.
- **Natural Language Aggregations**: Generates deterministic SQL queries (`SUM`, `AVG`, `MAX`, `MIN`, `COUNT`, `GROUP BY`) based on identified column candidates and schema metadata.

### D. Grounded QA & Strict Refusal (`generation/`)
- **Zero-Hallucination Policy**: If top retrieved RRF score is below confidence threshold or zero matching records are returned, the copilot issues an explicit refusal:
  > *"Insufficient context available in ingested enterprise documents or structured data to answer this question."*
- **Citation Rendering**: Every statement links directly to its source document and chunk index or SQL query executed.

### E. Production Interfaces (`api/` & `ui/`)
- **FastAPI REST Service**: Production endpoints for document upload, schema introspection, health monitoring, and query execution.
- **Streamlit Analytics Dashboard**: Real-time interactive inspection of retrieval ranks, SQL tables, and grounded answers.
