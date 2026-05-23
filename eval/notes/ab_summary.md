# Retrieve vs Retrieve-Rerank A/B Summary

## Result
Across 7 evaluation cases:
- retrieve top1 hit rate: 2/7
- retrieve_rerank top1 hit rate: 5/7
- repaired cases: 3
- worsened cases: 0

## Interpretation
The lightweight query-normalized lexical reranker substantially improved top1 ranking without harming any already-correct top1 case in this test set.

## Pattern
The reranker was especially helpful for:
- project purpose questions
- limitation / constraint questions
- method + explanation questions

It improved but did not fully fix:
- abstract PDF objective questions
- baseline-value boundary questions

## Takeaway
The reranker is strong enough to justify an A/B integration into the answer generation path, ideally through a separate experimental endpoint such as `/ask_rerank`.