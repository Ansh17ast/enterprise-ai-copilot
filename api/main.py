import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from ingestion.parser import DocumentParser
from ingestion.chunker import TextChunker, DocumentChunk
from retrieval.hybrid import HybridRetriever
from analytics.data_engine import DataEngine
from analytics.query_generator import AnalyticsQueryGenerator
from routing.router import QueryRouter, QueryIntent
from generation.grounded_qa import GroundedQASynthesizer
from api.schemas import (
    QueryRequest, QueryResponse, CitationItem,
    DocumentUploadResponse, TableSchemaResponse, HealthResponse
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise document intelligence, hybrid RAG retrieval, and structured analytics copilot."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Singletons
data_engine = DataEngine(db_path=settings.SQLITE_DB_PATH)
analytics_generator = AnalyticsQueryGenerator(data_engine)
hybrid_retriever = HybridRetriever(rrf_k=settings.RRF_K)
query_router = QueryRouter()
qa_synthesizer = GroundedQASynthesizer(api_key=settings.GEMINI_API_KEY)
chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
all_indexed_chunks: List[DocumentChunk] = []

def initialize_sample_data():
    global all_indexed_chunks
    samples_dir = settings.SAMPLES_DIR
    if not samples_dir.exists():
        return

    new_chunks = []
    for file_path in samples_dir.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() == ".csv":
                data_engine.load_csv(file_path)
            parsed = DocumentParser.parse_file(file_path)
            chunks = chunker.chunk_document(parsed)
            new_chunks.extend(chunks)

    all_indexed_chunks = new_chunks
    hybrid_retriever.index_chunks(all_indexed_chunks)

@app.on_event("startup")
def startup_event():
    initialize_sample_data()

@app.get(f"{settings.API_PREFIX}/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        indexed_chunks=len(all_indexed_chunks),
        registered_tables=len(data_engine.list_tables())
    )

@app.post(f"{settings.API_PREFIX}/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    global all_indexed_chunks
    try:
        content_bytes = await file.read()
        filename = file.filename or "uploaded_file.txt"
        ext = Path(filename).suffix.lower()

        # Temporary write to disk for parser
        temp_dir = settings.DATA_DIR / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / filename
        with open(temp_path, "wb") as f:
            f.write(content_bytes)

        if ext == ".csv":
            data_engine.load_csv(temp_path)

        parsed = DocumentParser.parse_file(temp_path)
        chunks = chunker.chunk_document(parsed)
        all_indexed_chunks.extend(chunks)
        hybrid_retriever.index_chunks(all_indexed_chunks)

        return DocumentUploadResponse(
            filename=filename,
            doc_id=parsed.doc_id,
            doc_type=parsed.doc_type,
            chunks_created=len(chunks),
            status="indexed_successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")

@app.get(f"{settings.API_PREFIX}/analytics/schema", response_model=TableSchemaResponse)
def get_analytics_schema():
    tables = data_engine.list_tables()
    info_list = []
    for t in tables:
        info = data_engine.get_table_info(t)
        if info:
            info_list.append(info.model_dump() if hasattr(info, "model_dump") else info.dict())
    return TableSchemaResponse(tables=info_list)

@app.post(f"{settings.API_PREFIX}/query", response_model=QueryResponse)
def process_query(req: QueryRequest):
    start_time = time.perf_counter()
    tables = data_engine.list_tables()

    # Determine execution route
    if req.force_route:
        try:
            intent = QueryIntent(req.force_route)
            routing = query_router.route_query(req.query, available_tables=tables)
            routing.intent = intent
        except ValueError:
            routing = query_router.route_query(req.query, available_tables=tables)
    else:
        routing = query_router.route_query(req.query, available_tables=tables)

    # Route execution
    if routing.intent == QueryIntent.STRUCTURED_ANALYTICS:
        analytics_result = analytics_generator.generate_and_execute(req.query)
        grounded = qa_synthesizer.synthesize_analytics_answer(req.query, analytics_result)

    elif routing.intent == QueryIntent.UNSTRUCTURED_RAG:
        search_results = hybrid_retriever.search(req.query, top_k=req.top_k)
        grounded = qa_synthesizer.synthesize_rag_answer(req.query, search_results, min_score=settings.MIN_RELEVANCE_SCORE)

    else:  # HYBRID_COMPOSITE
        analytics_result = analytics_generator.generate_and_execute(req.query)
        analytics_ans = qa_synthesizer.synthesize_analytics_answer(req.query, analytics_result)
        search_results = hybrid_retriever.search(req.query, top_k=req.top_k)
        rag_ans = qa_synthesizer.synthesize_rag_answer(req.query, search_results, min_score=settings.MIN_RELEVANCE_SCORE)
        grounded = qa_synthesizer.synthesize_hybrid_answer(req.query, rag_ans, analytics_ans)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    citation_items = [
        CitationItem(
            source=c.source,
            chunk_index=c.chunk_index,
            snippet=c.snippet,
            score=c.relevance_score
        ) for c in grounded.citations
    ]

    return QueryResponse(
        query=req.query,
        answer=grounded.answer,
        route_used=grounded.execution_route,
        confidence=routing.confidence,
        citations=citation_items,
        execution_time_ms=round(elapsed_ms, 2),
        is_refusal=grounded.is_refusal
    )
