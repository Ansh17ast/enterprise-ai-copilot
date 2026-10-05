import uuid
from typing import List, Dict
from pydantic import BaseModel, Field
from ingestion.parser import ParsedDocument

class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    doc_id: str
    source: str
    chunk_index: int
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        text = doc.content
        if not text:
            return []

        chunks: List[DocumentChunk] = []
        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If not at the end of text, attempt to break on paragraph or sentence boundary
            if end < text_len:
                last_newline = text.rfind("\n", start, end)
                last_period = text.rfind(". ", start, end)
                if last_newline != -1 and last_newline > start + (self.chunk_size // 2):
                    end = last_newline + 1
                elif last_period != -1 and last_period > start + (self.chunk_size // 2):
                    end = last_period + 2

            chunk_text = text[start:end].strip()
            if chunk_text:
                meta = {
                    "doc_id": doc.doc_id,
                    "source": doc.filename,
                    "doc_type": doc.doc_type,
                    "char_start": str(start),
                    "char_end": str(end),
                    "chunk_index": str(chunk_idx)
                }
                meta.update(doc.metadata)

                chunks.append(DocumentChunk(
                    doc_id=doc.doc_id,
                    source=doc.filename,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    metadata=meta
                ))
                chunk_idx += 1

            if end >= text_len:
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks
