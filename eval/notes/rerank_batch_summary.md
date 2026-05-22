# Rerank Batch Summary

## Cases
Q04, Q05, Q01, H03, H06

## Result
- top1 before rerank: 0/5
- top1 after rerank: 3/5
- promoted cases: 5/5
- worsened cases: 0/5

## Interpretation
Lightweight query-normalized lexical reranking consistently improved ranking for all tested non-top1 cases.
It was especially effective for markdown purpose/limitation questions and one PDF methods question.
For more abstract PDF objective/baseline questions, reranking improved rank but did not always reach top1.

## Takeaway
This rerank strategy is promising enough to justify an A/B integration into the retrieval path.