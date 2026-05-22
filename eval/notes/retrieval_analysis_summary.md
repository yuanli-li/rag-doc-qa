# Retrieval Analysis Summary

## Metrics
- gold_in_top_k_rate: 100% (10/10)
- top1_hit_rate: 50% (5/10)
- ask_rescue_rate: 100% (5/5 among non-top1 cases)
- rerank_need_rate: 50% (5/10)

## Interpretation
Current retrieval recall is strong: gold evidence entered top_k in all tested cases.
The main weakness is ranking rather than recall, since gold was top1 in only half of the cases.
The ask stage consistently rescued non-top1 retrieval errors by citing the correct gold evidence and still producing correct final answers.
This suggests that the next likely improvement target is reranking or stronger retrieval ranking, rather than answer grounding.