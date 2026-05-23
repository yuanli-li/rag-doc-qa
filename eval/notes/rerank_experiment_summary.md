# Minimal Rerank Experiment Summary

## Setup
We tested a lightweight reranking strategy on two representative top1 failure cases:
- T01: PDF objective / experimental goal question
- T05: Markdown main purpose question

The rerank score was based on:
- negative cosine distance
- lexical overlap bonus
- lightweight query normalization for abstract intent terms such as:
  - "experimental goal" -> "objective"
  - "built around" -> "objective"
  - "main purpose" -> "purpose"

## Result
Before reranking:
- T01 gold rank = 2
- T05 gold rank = 2

After normalized lexical reranking:
- T01 gold rank = 1
- T05 gold rank = 1

## Interpretation
A naive lexical reranker was not sufficient, because it could over-reward superficially related chunks.
However, adding lightweight query normalization made reranking more semantically aligned and successfully promoted the true gold evidence to top1 in both tested failure cases.

## Takeaway
The current retrieval weakness appears to be a ranking problem for abstract purpose/objective-style paraphrase questions.
A lightweight reranking layer with query normalization may be a practical next step before introducing a heavier learned reranker.


