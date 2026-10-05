import json
import time
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from typing import List, Dict, Any
import numpy as np

from core.config import settings
from ingestion.parser import DocumentParser
from ingestion.chunker import TextChunker
from retrieval.hybrid import HybridRetriever
from analytics.data_engine import DataEngine
from analytics.query_generator import AnalyticsQueryGenerator
from routing.router import QueryRouter, QueryIntent
from generation.grounded_qa import GroundedQASynthesizer

def run_evaluation():
    print("=" * 60)
    print("RUNNING ENTERPRISE AI COPILOT EVALUATION HARNESS")
    print("=" * 60)

    # 1. Initialize Components
    db_path = settings.DATA_DIR / "eval_analytics.db"
    if db_path.exists():
        db_path.unlink()

    engine = DataEngine(db_path=db_path)
    analytics_gen = AnalyticsQueryGenerator(engine)
    retriever = HybridRetriever(rrf_k=settings.RRF_K)
    router = QueryRouter()
    synthesizer = GroundedQASynthesizer(api_key=settings.GEMINI_API_KEY)
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    # 2. Ingest Sample Data
    chunks = []
    for p in settings.SAMPLES_DIR.iterdir():
        if p.is_file():
            if p.suffix.lower() == ".csv":
                engine.load_csv(p)
            parsed = DocumentParser.parse_file(p)
            chunks.extend(chunker.chunk_document(parsed))
    retriever.index_chunks(chunks)

    print(f"Indexed {len(chunks)} document chunks across {len(list(settings.SAMPLES_DIR.iterdir()))} files.")
    print(f"Loaded {len(engine.list_tables())} structured tables.\n")

    # 3. Load Evaluation Dataset
    dataset_path = BASE_DIR / "evaluation" / "dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    latencies = []
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks = []
    router_correct = 0
    refusal_correct = 0
    total_rag = 0
    total_refusal = 0

    for tc in test_cases:
        qid = tc["id"]
        query = tc["query"]
        cat = tc["category"]
        expected_src = tc["expected_source"]
        expected_intent = tc["expected_intent"]

        t0 = time.perf_counter()
        routing = router.route_query(query, available_tables=engine.list_tables())

        # Measure routing
        is_router_match = (routing.intent.value == expected_intent)
        if is_router_match:
            router_correct += 1

        # Execute
        if routing.intent == QueryIntent.STRUCTURED_ANALYTICS:
            analytics_res = analytics_gen.generate_and_execute(query)
            ans = synthesizer.synthesize_analytics_answer(query, analytics_res)
            retrieval_hits = []
        elif routing.intent == QueryIntent.UNSTRUCTURED_RAG:
            retrieval_hits = retriever.search(query, top_k=4)
            ans = synthesizer.synthesize_rag_answer(query, retrieval_hits)
        else:
            analytics_res = analytics_gen.generate_and_execute(query)
            ans_a = synthesizer.synthesize_analytics_answer(query, analytics_res)
            retrieval_hits = retriever.search(query, top_k=4)
            ans_r = synthesizer.synthesize_rag_answer(query, retrieval_hits)
            ans = synthesizer.synthesize_hybrid_answer(query, ans_r, ans_a)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # Evaluate Retrieval Metrics (for RAG queries)
        if cat == "unstructured_rag":
            total_rag += 1
            sources = [h.chunk.metadata.get("source", "") for h in retrieval_hits]
            if sources and sources[0] == expected_src:
                hit_at_1 += 1
            if expected_src in sources[:3]:
                hit_at_3 += 1
            # Reciprocal rank
            rr = 0.0
            for rank, s in enumerate(sources, start=1):
                if s == expected_src:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        # Evaluate Refusal
        if cat == "out_of_domain_refusal":
            total_refusal += 1
            if ans.is_refusal:
                refusal_correct += 1

        results.append({
            "id": qid,
            "query": query,
            "category": cat,
            "route_used": routing.intent.value,
            "is_refusal": ans.is_refusal,
            "latency_ms": round(elapsed_ms, 2),
            "citations_count": len(ans.citations)
        })

    # Summary Metrics
    hit_rate_1 = (hit_at_1 / total_rag) * 100.0 if total_rag else 0.0
    hit_rate_3 = (hit_at_3 / total_rag) * 100.0 if total_rag else 0.0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    router_acc = (router_correct / len(test_cases)) * 100.0
    refusal_acc = (refusal_correct / total_refusal) * 100.0 if total_refusal else 0.0
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    mean_latency = np.mean(latencies)

    print(f"Evaluation Complete across {len(test_cases)} test queries:")
    print(f"- Router Intent Accuracy: {router_acc:.1f}%")
    print(f"- Retrieval Hit Rate @ 1: {hit_rate_1:.1f}%")
    print(f"- Retrieval Hit Rate @ 3: {hit_rate_3:.1f}%")
    print(f"- Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"- Out-of-Domain Refusal Accuracy: {refusal_acc:.1f}%")
    print(f"- Latency p50: {p50_latency:.2f} ms")
    print(f"- Latency p95: {p95_latency:.2f} ms")
    print(f"- Latency Mean: {mean_latency:.2f} ms")

    # Generate benchmark report markdown
    report_path = BASE_DIR / "evaluation" / "benchmark_report.md"
    report_content = f"""# Enterprise AI Copilot - Benchmark Evaluation Report

## Benchmark Summary Metrics
- **Evaluated Test Cases**: {len(test_cases)}
- **Intent Routing Accuracy**: {router_acc:.1f}%
- **Retrieval Hit Rate @ 1**: {hit_rate_1:.1f}%
- **Retrieval Hit Rate @ 3**: {hit_rate_3:.1f}%
- **Mean Reciprocal Rank (MRR)**: {mrr:.4f}
- **Out-of-Domain Grounding Refusal Accuracy**: {refusal_acc:.1f}%
- **Query Turnaround Latency (p50)**: {p50_latency:.2f} ms
- **Query Turnaround Latency (p95)**: {p95_latency:.2f} ms
- **Query Turnaround Latency (Mean)**: {mean_latency:.2f} ms

## Per-Query Execution Log
| ID | Category | Intent Route | Refusal? | Citations | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in results:
        report_content += f"| `{r['id']}` | {r['category']} | `{r['route_used']}` | {r['is_refusal']} | {r['citations_count']} | {r['latency_ms']} ms |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nReport saved to: {report_path}")
    return {
        "hit_rate_1": hit_rate_1,
        "hit_rate_3": hit_rate_3,
        "mrr": mrr,
        "router_acc": router_acc,
        "refusal_acc": refusal_acc,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "mean_latency": mean_latency
    }

if __name__ == "__main__":
    run_evaluation()
