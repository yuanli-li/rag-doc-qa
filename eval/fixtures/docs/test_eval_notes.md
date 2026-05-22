# Project Overview

This markdown document was created for testing a retrieval and grounded question-answering pipeline. The purpose of the project is to provide a compact, section-structured source for evaluating section-aware retrieval.

## Why This Project

Many small RAG systems can ingest markdown easily, but they still need to prove that they can retrieve the correct section and refuse unsupported claims. This file is intentionally simple so that section metadata can be tested clearly.

## Workflow

The current workflow has four stages: ingest the document, split it into chunks, retrieve relevant evidence, and answer only when the evidence is sufficient.

## Current Limitations

The current system does not include OCR for scanned PDFs. It also does not include a learned reranker, and it does not report benchmark metrics such as F1, MRR, or accuracy in this markdown file.

## Architecture Decisions

The project keeps document-level records separate from chunk-level records. It also stores lightweight metadata so that answers can cite source sections or pages.

## Future Work

Future iterations may improve PDF layout parsing, add stronger refusal logic, and support more detailed evaluation workflows.
