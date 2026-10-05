import pytest
from pathlib import Path
from ingestion.parser import DocumentParser, ParsedDocument
from ingestion.chunker import TextChunker

def test_parse_text():
    raw = "Global Enterprise Security Policy.\nSection 1: Scope."
    doc = DocumentParser.parse_text(raw, "policy.txt")
    assert doc.filename == "policy.txt"
    assert doc.doc_type == "txt"
    assert "Global Enterprise Security" in doc.content

def test_chunker_sliding_window():
    text = "Sentence one. " * 30  # ~420 chars
    doc = ParsedDocument(filename="sample.txt", doc_type="txt", content=text)
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) > 1
    for c in chunks:
        assert c.source == "sample.txt"
        assert len(c.text) <= 160
        assert "chunk_index" in c.metadata
