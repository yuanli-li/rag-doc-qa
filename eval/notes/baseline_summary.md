# Baseline Summary

## Result
Initial 6-question baseline achieved 6/6 correctness with 6/6 citation match and no hallucinations.

## Strengths
- Markdown and PDF ingest both work.
- Ask can select correct evidence even when retrieve top1 is not ideal.
- Grounded negative answers work for explicitly absent information.

## Main remaining issue
- Retrieval ranking is not always optimal at top1, especially for some objective/limitation questions.