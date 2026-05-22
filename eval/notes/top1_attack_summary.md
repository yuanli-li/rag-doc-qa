# Top1 Attack Summary

## Metrics
- gold_in_top_k_rate: 100% (6/6)
- top1_hit_rate: 66.7% (4/6)
- rerank_need_rate: 33.3% (2/6)

## Pattern
The main ranking weakness appears in purpose/objective-style paraphrase questions.
When the question asks for the main goal or main purpose, top1 may be captured by a semantically related but less direct chunk.
In contrast, direct evidence-seeking questions (measurement sentence, citation-friendly design evidence, explicit negative benchmark statement) are much more stable at top1.

## Interpretation
Retrieval recall remains strong, but ranking is selectively weaker for abstract, summary-like intent questions.
This suggests future reranking or query reformulation efforts should focus first on purpose/objective-style prompts.