from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What is our mandatory retention period for financial logs?"])
    top_k: int = Field(default=4, ge=1, le=20)
    force_route: Optional[str] = Field(default=None, examples=["unstructured_rag"])

class CitationItem(BaseModel):
    source: str
    chunk_index: Optional[str] = None
    snippet: str
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    route_used: str
    confidence: float
    citations: List[CitationItem]
    execution_time_ms: float
    is_refusal: bool

class DocumentUploadResponse(BaseModel):
    filename: str
    doc_id: str
    doc_type: str
    chunks_created: int
    status: str

class TableSchemaResponse(BaseModel):
    tables: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    version: str
    indexed_chunks: int
    registered_tables: int
