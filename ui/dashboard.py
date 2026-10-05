import streamlit as st
import time
from pathlib import Path
from core.config import settings
from ingestion.parser import DocumentParser
from ingestion.chunker import TextChunker
from retrieval.hybrid import HybridRetriever
from analytics.data_engine import DataEngine
from analytics.query_generator import AnalyticsQueryGenerator
from routing.router import QueryRouter, QueryIntent
from generation.grounded_qa import GroundedQASynthesizer

st.set_page_config(
    page_title="Enterprise AI Analytics & Document Copilot",
    page_icon="🏢",
    layout="wide"
)

@st.cache_resource
def load_system():
    engine = DataEngine(db_path=settings.SQLITE_DB_PATH)
    analytics_gen = AnalyticsQueryGenerator(engine)
    retriever = HybridRetriever(rrf_k=settings.RRF_K)
    router = QueryRouter()
    synthesizer = GroundedQASynthesizer(api_key=settings.GEMINI_API_KEY)
    chunker_inst = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    # Ingest samples
    chunks = []
    if settings.SAMPLES_DIR.exists():
        for p in settings.SAMPLES_DIR.iterdir():
            if p.is_file():
                if p.suffix.lower() == ".csv":
                    engine.load_csv(p)
                parsed = DocumentParser.parse_file(p)
                chunks.extend(chunker_inst.chunk_document(parsed))
    retriever.index_chunks(chunks)
    return engine, analytics_gen, retriever, router, synthesizer, chunker_inst, chunks

engine, analytics_gen, retriever, router, synthesizer, chunker_inst, all_chunks = load_system()

st.title("🏢 Enterprise AI Analytics & Document Intelligence Copilot")
st.markdown(
    "Unified hybrid RAG and deterministic structured analytics engine. Grounded answers, explicit citations, and zero-hallucination guardrails."
)

# Sidebar
with st.sidebar:
    st.header("📊 System Health & Stats")
    st.metric(label="Indexed Document Chunks", value=len(all_chunks))
    st.metric(label="Structured Tables", value=len(engine.list_tables()))
    st.caption("Active Engine: BM25 + Vector RRF | SQLite Analytical Core")
    st.divider()
    st.subheader("Loaded Tables")
    for t in engine.list_tables():
        st.write(f"- `{t}`")

tabs = st.tabs(["💬 Ask Copilot", "📈 Structured Data Analytics", "🔍 Hybrid Search Inspector", "📁 Document Ingestion"])

# Tab 1: Ask Copilot
with tabs[0]:
    st.subheader("Query the Enterprise Knowledge Base")
    sample_queries = [
        "What is our mandatory retention period for financial logs?",
        "What is the total revenue by region?",
        "What are the MFA requirements for production systems?",
        "What are the SLA uptime commitments and financial penalties for vendors?",
        "What is the average operating margin by department?",
        "What is the policy for planetary colonization?" # Test refusal
    ]
    selected_sample = st.selectbox("Sample Enterprise Questions:", [""] + sample_queries)
    user_query = st.text_input("Or enter your question:", value=selected_sample)

    if st.button("Submit Query", type="primary") and user_query:
        t0 = time.perf_counter()
        routing = router.route_query(user_query, available_tables=engine.list_tables())

        col1, col2 = st.columns([3, 1])
        with col2:
            st.info(f"**Route**: `{routing.intent.value}`\n\n**Confidence**: `{routing.confidence * 100:.1f}%`")

        if routing.intent == QueryIntent.STRUCTURED_ANALYTICS:
            analytics_res = analytics_gen.generate_and_execute(user_query)
            ans = synthesizer.synthesize_analytics_answer(user_query, analytics_res)
        elif routing.intent == QueryIntent.UNSTRUCTURED_RAG:
            res = retriever.search(user_query, top_k=4)
            ans = synthesizer.synthesize_rag_answer(user_query, res)
        else:
            analytics_res = analytics_gen.generate_and_execute(user_query)
            ans_a = synthesizer.synthesize_analytics_answer(user_query, analytics_res)
            res = retriever.search(user_query, top_k=4)
            ans_r = synthesizer.synthesize_rag_answer(user_query, res)
            ans = synthesizer.synthesize_hybrid_answer(user_query, ans_r, ans_a)

        elapsed = (time.perf_counter() - t0) * 1000.0

        with col1:
            if ans.is_refusal:
                st.warning(f"⚠️ **Refusal Guardrail Triggered**:\n\n{ans.answer}")
            else:
                st.markdown(ans.answer)
            st.caption(f"⏱️ Turnaround Latency: {elapsed:.2f} ms")

        if ans.citations:
            with st.expander(f"📚 Cited Sources ({len(ans.citations)})", expanded=True):
                for c in ans.citations:
                    st.markdown(f"- **Source**: `{c.source}` (Chunk: `{c.chunk_index}`) | Score: `{c.relevance_score}`")
                    st.caption(f"> {c.snippet}")

# Tab 2: Structured Data Analytics
with tabs[1]:
    st.subheader("Structured Database & Analytics Explorer")
    tables = engine.list_tables()
    if tables:
        chosen_table = st.selectbox("Select Table:", tables)
        info = engine.get_table_info(chosen_table)
        if info:
            st.write(f"**Total Records**: {info.row_count} | **Columns**: {', '.join(info.columns)}")
            custom_sql = st.text_area("Execute Read-Only SQL Query:", value=f"SELECT * FROM {chosen_table} LIMIT 5")
            if st.button("Run SQL"):
                try:
                    res = engine.execute_query(custom_sql)
                    st.dataframe(res["data"])
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")

# Tab 3: Hybrid Search Inspector
with tabs[2]:
    st.subheader("Hybrid Retrieval Fusion Breakdown")
    search_q = st.text_input("Test Retrieval Query:", value="retention SOX guidelines")
    if st.button("Inspect Hybrid Ranking") and search_q:
        results = retriever.search(search_q, top_k=5)
        for r in results:
            st.markdown(f"**Chunk**: `{r.chunk.metadata.get('source')}` (Index: `{r.chunk.metadata.get('chunk_index')}`)")
            st.write(f"- **RRF Fused Score**: `{r.rrf_score}` | Vector Rank: `{r.vector_rank}` | BM25 Rank: `{r.bm25_rank}`")
            st.text(r.chunk.text[:200] + "...")
            st.divider()

# Tab 4: Document Ingestion
with tabs[3]:
    st.subheader("Upload Document or Dataset")
    uploaded = st.file_uploader("Upload TXT, MD, CSV, or PDF", type=["txt", "md", "csv", "pdf"])
    if uploaded and st.button("Process & Index"):
        content = uploaded.read()
        p = settings.DATA_DIR / "uploads" / uploaded.name
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(content)
        if p.suffix.lower() == ".csv":
            engine.load_csv(p)
        doc = DocumentParser.parse_file(p)
        new_c = chunker_inst.chunk_document(doc)
        all_chunks.extend(new_c)
        retriever.index_chunks(all_chunks)
        st.success(f"Indexed {len(new_c)} new chunks from {uploaded.name}!")
