import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from retrieval.hybrid import HybridSearchResult

class Citation(BaseModel):
    source: str
    chunk_index: Optional[str] = None
    snippet: str
    relevance_score: float

class GroundedAnswer(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    is_refusal: bool = False
    execution_route: str

class GroundedQASynthesizer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def synthesize_rag_answer(self, query: str, results: List[HybridSearchResult], min_score: float = 0.020) -> GroundedAnswer:
        # Check if we have valid results
        if not results:
            return GroundedAnswer(
                query=query,
                answer="Insufficient context available in the ingested enterprise documents to answer this question factually. No matching evidence was retrieved.",
                citations=[],
                is_refusal=True,
                execution_route="unstructured_rag"
            )

        top_hit = results[0]
        # Strict Grounding Guard:
        # A document must have genuine semantic relevance (cosine similarity >= 0.30).
        # Lexical-only matches on generic keywords (e.g. 'policy') without semantic grounding are rejected.
        is_semantically_relevant = (top_hit.vector_score is not None and top_hit.vector_score >= 0.30)
        has_sufficient_rrf = (top_hit.rrf_score >= min_score)

        if not is_semantically_relevant or not has_sufficient_rrf:
            return GroundedAnswer(
                query=query,
                answer="Insufficient context available in the ingested enterprise documents to answer this question factually. No high-confidence evidence was retrieved.",
                citations=[],
                is_refusal=True,
                execution_route="unstructured_rag"
            )

        citations: List[Citation] = []
        context_snippets: List[str] = []

        for res in results[:3]:
            meta = res.chunk.metadata
            c = Citation(
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", "0"),
                snippet=res.chunk.text[:250] + ("..." if len(res.chunk.text) > 250 else ""),
                relevance_score=res.rrf_score
            )
            citations.append(c)
            context_snippets.append(f"[{c.source} | Chunk {c.chunk_index}]: {res.chunk.text}")

        # If Gemini API key is configured, invoke LLM; otherwise use grounded extraction
        if self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                prompt = (
                    f"You are an enterprise document intelligence copilot. Answer the question using ONLY the provided evidence.\n"
                    f"If the evidence does not clearly contain the answer, state that there is insufficient context.\n"
                    f"Include explicit inline citations like [source_name, Chunk #] in your response.\n\n"
                    f"Evidence:\n" + "\n\n".join(context_snippets) + f"\n\nQuestion: {query}\n\nAnswer:"
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                answer_text = response.text.strip()
                return GroundedAnswer(
                    query=query,
                    answer=answer_text,
                    citations=citations,
                    is_refusal=False,
                    execution_route="unstructured_rag"
                )
            except Exception:
                pass  # Fall through to deterministic synthesizer

        # Deterministic Grounded Synthesis
        primary_evidence = results[0].chunk.text
        primary_source = citations[0].source
        primary_chunk = citations[0].chunk_index
        answer_text = (
            f"Based on enterprise documentation [{primary_source}, Chunk #{primary_chunk}]:\n\n"
            f"{primary_evidence}\n\n"
            f"Additional contextual evidence was referenced from {len(citations) - 1} supporting passage(s)."
        )
        return GroundedAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            is_refusal=False,
            execution_route="unstructured_rag"
        )

    def synthesize_analytics_answer(self, query: str, analytics_result: Dict[str, Any]) -> GroundedAnswer:
        if "error" in analytics_result:
            return GroundedAnswer(
                query=query,
                answer=f"Could not compute analytics: {analytics_result['error']}",
                citations=[],
                is_refusal=True,
                execution_route="structured_analytics"
            )

        data = analytics_result.get("data", [])
        sql = analytics_result.get("query", "")
        table = analytics_result.get("table_used", "data_table")

        if not data:
            return GroundedAnswer(
                query=query,
                answer=f"The structured query returned zero matching records from table '{table}'.",
                citations=[],
                is_refusal=True,
                execution_route="structured_analytics"
            )

        citation = Citation(
            source=f"table:{table}",
            chunk_index="sql_engine",
            snippet=f"Executed SQL: {sql} | Rows: {len(data)}",
            relevance_score=1.0
        )

        # Format summary table
        if len(data) == 1 and len(data[0]) == 1:
            k, v = list(data[0].items())[0]
            answer_text = f"Calculated {k.replace('_', ' ')} is **{v:,}** (Source: table '{table}')."
        else:
            lines = [f"Structured analysis from table `{table}` (Query: `{sql}`):", ""]
            for idx, row in enumerate(data[:10], start=1):
                row_str = " | ".join(f"{k}: **{v}**" for k, v in row.items())
                lines.append(f"{idx}. {row_str}")
            answer_text = "\n".join(lines)

        return GroundedAnswer(
            query=query,
            answer=answer_text,
            citations=[citation],
            is_refusal=False,
            execution_route="structured_analytics"
        )

    def synthesize_hybrid_answer(self, query: str, rag_answer: GroundedAnswer, analytics_answer: GroundedAnswer) -> GroundedAnswer:
        combined_citations = analytics_answer.citations + rag_answer.citations
        combined_text = (
            f"### Quantitative Analytics Summary\n{analytics_answer.answer}\n\n"
            f"### Document Context & Policy Guidelines\n{rag_answer.answer}"
        )
        return GroundedAnswer(
            query=query,
            answer=combined_text,
            citations=combined_citations,
            is_refusal=rag_answer.is_refusal and analytics_answer.is_refusal,
            execution_route="hybrid_composite"
        )
