# Final Evaluation Summary

## System status
The current RAG system supports:
- document ingestion for markdown and PDF
- vector retrieval with pgvector
- grounded QA with citation extraction
- refusal logic based on similarity thresholds
- optional lightweight reranking at retrieval time

## Baseline result
The initial baseline set showed that the end-to-end pipeline was working correctly on both markdown and PDF test documents.

## Retrieval analysis
Across the earlier retrieval analysis set:
- gold evidence entered top_k reliably
- the main weakness was ranking rather than recall
- purpose/objective-style paraphrase questions were more likely to miss at top1

## Retrieve vs Retrieve-Rerank A/B
On a focused 7-case A/B comparison:
- retrieve top1 hit rate: 2/7
- retrieve_rerank top1 hit rate: 5/7
- repaired cases: 3
- worsened cases: 0

This shows that the lightweight query-normalized lexical reranker substantially improves top1 ranking without harming already-correct top1 cases in the tested subset.

## Ask vs Ask-Rerank A/B
On a 5-case downstream QA comparison:
- ask gold-citation hits: 5/5
- ask_rerank gold-citation hits: 5/5
- improved cases: 0
- worsened cases: 0

Interpretation:
The current ask pipeline already recovers gold evidence reliably from the retrieved top_k set.
As a result, retrieval reranking improves evidence ordering but does not yet improve final QA metrics on this small ask-level test set.

## Conclusion
The reranker is worth keeping as a retrieval-layer enhancement, especially for abstract purpose/objective-style queries.
At the current stage, it is best treated as an experimental retrieval improvement rather than a default replacement for the QA path.

## Practical decision
Recommended default endpoints:
- keep `/ask` as the default QA endpoint
- keep `/retrieve_rerank` as the experimental improved retrieval endpoint
- keep `/ask_rerank` only for further experiments, not as the default