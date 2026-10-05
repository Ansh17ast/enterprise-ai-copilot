# Enterprise AI Copilot - Benchmark Evaluation Report

## Benchmark Summary Metrics
- **Evaluated Test Cases**: 10
- **Intent Routing Accuracy**: 100.0%
- **Retrieval Hit Rate @ 1**: 100.0%
- **Retrieval Hit Rate @ 3**: 100.0%
- **Mean Reciprocal Rank (MRR)**: 1.0000
- **Out-of-Domain Grounding Refusal Accuracy**: 100.0%
- **Query Turnaround Latency (p50)**: 12.05 ms
- **Query Turnaround Latency (p95)**: 14.45 ms
- **Query Turnaround Latency (Mean)**: 9.84 ms

## Per-Query Execution Log
| ID | Category | Intent Route | Refusal? | Citations | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `eval_01` | unstructured_rag | `unstructured_rag` | False | 3 | 13.93 ms |
| `eval_02` | unstructured_rag | `unstructured_rag` | False | 3 | 12.36 ms |
| `eval_03` | unstructured_rag | `unstructured_rag` | False | 3 | 11.75 ms |
| `eval_04` | unstructured_rag | `unstructured_rag` | False | 3 | 12.78 ms |
| `eval_05` | structured_analytics | `structured_analytics` | False | 1 | 3.98 ms |
| `eval_06` | structured_analytics | `structured_analytics` | False | 1 | 2.0 ms |
| `eval_07` | structured_analytics | `structured_analytics` | False | 1 | 1.72 ms |
| `eval_08` | out_of_domain_refusal | `unstructured_rag` | True | 0 | 13.38 ms |
| `eval_09` | out_of_domain_refusal | `unstructured_rag` | True | 0 | 11.6 ms |
| `eval_10` | hybrid_composite | `hybrid_composite` | False | 4 | 14.87 ms |
