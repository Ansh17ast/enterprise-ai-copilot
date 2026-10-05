import csv
import io
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ParsedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    filename: str
    doc_type: str  # pdf, csv, txt, md
    content: str
    metadata: Dict[str, str] = Field(default_factory=dict)
    raw_tables: Optional[List[Dict[str, str]]] = None

class DocumentParser:
    @staticmethod
    def parse_text(text: str, filename: str) -> ParsedDocument:
        ext = Path(filename).suffix.lower().lstrip(".") or "txt"
        return ParsedDocument(
            filename=filename,
            doc_type=ext,
            content=text.strip(),
            metadata={"source": filename, "char_count": str(len(text))}
        )

    @staticmethod
    def parse_file(file_path: Path) -> ParsedDocument:
        ext = file_path.suffix.lower()
        filename = file_path.name

        if ext in [".txt", ".md", ".log"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return ParsedDocument(
                filename=filename,
                doc_type=ext.lstrip("."),
                content=content.strip(),
                metadata={"source": filename, "path": str(file_path)}
            )

        elif ext == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                pages_text = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pages_text.append(f"--- Page {page_num + 1} ---\n" + page.get_text())
                doc.close()
                combined_text = "\n\n".join(pages_text)
                return ParsedDocument(
                    filename=filename,
                    doc_type="pdf",
                    content=combined_text.strip(),
                    metadata={"source": filename, "pages": str(len(pages_text))}
                )
            except Exception as e:
                return ParsedDocument(
                    filename=filename,
                    doc_type="pdf",
                    content=f"Error extracting PDF text: {str(e)}",
                    metadata={"error": str(e)}
                )

        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Format text representation for retrieval
            headers = reader.fieldnames or []
            summary_lines = [f"Table: {filename} with columns: {', '.join(headers)}", f"Total records: {len(rows)}"]
            for i, r in enumerate(rows[:50]):  # Sample up to 50 rows in text summary
                summary_lines.append(f"Row {i+1}: " + ", ".join(f"{k}={v}" for k, v in r.items()))
            
            return ParsedDocument(
                filename=filename,
                doc_type="csv",
                content="\n".join(summary_lines),
                metadata={"source": filename, "columns": ", ".join(headers), "row_count": str(len(rows))},
                raw_tables=rows
            )
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return ParsedDocument(
                filename=filename,
                doc_type="unknown",
                content=content.strip(),
                metadata={"source": filename}
            )
