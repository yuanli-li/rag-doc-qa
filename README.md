
# RAG Doc QA
RAG Doc QA is an interpretable RAG experimentation platform for technical documents. It lets users ingest markdown and PDF files, retrieve evidence, answer questions with chunk-level citations, and refuse unsupported claims instead of bluffing. Unlike a typical “chat with docs” demo, it makes retrieval, ranking, grounding, and refusal behavior observable and measurable, and includes an experimental reranking path for improving evidence ordering.

A minimal retrieval-augmented question answering (RAG) system for document ingestion, vector retrieval, grounded answering, citation extraction, refusal handling, and retrieval reranking experiments.

This project is designed to be small enough to understand end to end, while still including several practical components that matter in real systems:

- document ingestion for markdown and PDF
- chunking with metadata preservation
- vector similarity retrieval with PostgreSQL + pgvector
- grounded QA with cited evidence
- refusal logic based on retrieval similarity
- lightweight experimental retrieval reranking
- evaluation workflows for retrieval and QA behavior

---

## Why This Project

Many small RAG demos can answer questions over documents, but they often make it hard to inspect *why* an answer was produced, whether the answer is grounded in retrieved evidence, and whether the retrieval stage itself is doing the right thing.

This project was built to make those layers visible and testable.

The core goals are:

- keep the full pipeline easy to inspect
- support grounded answers with explicit chunk citations
- separate retrieval quality from answer-generation quality
- make refusal behavior measurable
- provide a clean sandbox for experiments such as reranking

---

## What the System Does

At a high level, the system supports three main workflows:

### 1. Ingest
A document is uploaded, parsed, split into chunks, embedded, and stored in PostgreSQL / pgvector.

### 2. Retrieve
A question is embedded and matched against stored chunks using vector similarity.

### 3. Ask
The system retrieves evidence, filters weak matches, optionally refuses unsupported questions, generates a grounded answer, and returns only the chunks actually cited by the answer.

---

## Current Features

- Markdown and text ingestion
- PDF ingestion with PyMuPDF-based parsing
- Chunk metadata such as:
  - page
  - section
  - heading level
  - source type
  - filename
- pgvector-backed similarity retrieval
- grounded QA with chunk-level citations like `[chunk_id]`
- refusal logic for weak evidence
- intent gate for conclusion-style questions
- optional retrieval reranking
- retrieval-only and QA-level evaluation utilities

---

## Project Structure

```text
rag-doc-qa/
├─ backend/
│  └─ app/
│     ├─ main.py
│     ├─ db.py
│     ├─ store.py
│     ├─ parsers.py
│     ├─ embeddings.py
│     ├─ retrieval.py
│     └─ ...
├─ infra/
│  ├─ docker-compose.yml
│  └─ db/
│     ├─ init.sql
│     └─ schema.sql
├─ eval/
│  ├─ fixtures/
│  │  └─ docs/
│  ├─ data/
│  │  └─ results/
│  ├─ notes/
│  ├─ run_retrieve_ab.py
│  ├─ run_ask_ab.py
│  └─ rerank_experiment.py
└─ README.md
```

---

## Architecture Overview

The system can be viewed as four layers:

### 1. Parsing and chunking
Uploaded files are parsed into structured text units and then chunked while preserving useful metadata.

### 2. Embedding and storage
Chunks are embedded and stored in PostgreSQL with pgvector.

### 3. Retrieval
A query is embedded and matched against stored chunks using cosine distance.

### 4. Grounded QA
The system selects evidence, optionally refuses weak questions, generates an answer, and returns cited chunks.

---

## Data Model

The project uses two main database tables:

### `documents`
Stores document-level metadata such as:
- title
- filename
- mime type
- sha256 digest

### `chunks`
Stores chunk-level data such as:
- `document_id`
- `chunk_index`
- `content`
- `metadata`
- `embedding`

This separation is intentional:
- document-level records preserve document identity
- chunk-level records support retrieval and citation
- lightweight metadata helps map answers back to sections or pages

---

## Supported Input Types

### Markdown / Text
Markdown is parsed into units based on headings. Metadata includes:
- `section`
- `level`

### PDF
PDFs are parsed with PyMuPDF when available. The parser:
- reads text blocks
- applies simple two-column-aware ordering
- cleans text
- splits content into paragraph-like units

PDF metadata includes:
- `page`

---

## Ingestion Pipeline

The ingestion flow is:

1. read uploaded file bytes
2. compute document hash
3. parse content into units
4. chunk units into overlapping text windows
5. embed chunk text
6. store document metadata
7. store chunk text, metadata, and embeddings

This enables later retrieval and grounded QA.

---

## Retrieval Pipeline

The basic retrieval pipeline is:

1. embed query text
2. run vector similarity search against chunk embeddings
3. return top-k results ordered by cosine distance

Each retrieval result includes:
- `chunk_id`
- `chunk_index`
- `filename`
- `source_type`
- `page` or `section`
- `cosine_distance`
- `snippet`

---

## Grounded QA Pipeline

The default `/ask` endpoint performs the following steps:

1. embed query
2. retrieve `raw_k` candidates
3. compute best similarity distance
4. refuse if the strongest evidence is too weak
5. filter by similarity threshold
6. select top-k filtered evidence
7. optionally apply an intent gate for conclusion-style questions
8. generate grounded answer
9. extract cited chunk IDs
10. return only chunks actually cited in the answer

This design helps separate:
- retrieval quality
- answer generation
- citation grounding

---

## Experimental Retrieval Reranking

This project includes an experimental retrieval reranking path exposed through `/retrieve_rerank`.

The reranker works as a lightweight post-retrieval layer:

1. retrieve top-k candidate chunks using vector similarity
2. apply lightweight query normalization
3. add lexical overlap and weighted keyword bonuses
4. reorder the retrieved candidates without changing their original cosine distance values

The goal of this reranking step is to improve evidence ordering, especially for abstract paraphrase-style queries such as:

- purpose / objective questions
- limitation / constraint questions
- method questions with explanatory phrasing

In evaluation, this reranker improved retrieval top-1 ranking on a focused failure-heavy subset. It was especially helpful when the correct evidence was already present in top-k but was not ranked first by the original vector retrieval.

However, on the current small ask-level A/B test set, this reranking improvement did not translate into higher downstream QA accuracy, because the existing `/ask` pipeline was already strong at recovering the gold evidence from the retrieved top-k set.

As a result, reranking is currently retained as an **experimental retrieval enhancement**, rather than the default QA path.

---

## API Endpoints

### `POST /ingest`
Ingests a document into the system.

Supported file types:
- `.pdf`
- `.md`
- `.txt`

What it does:
1. reads the uploaded file
2. parses the content into units
3. chunks the units
4. generates embeddings
5. stores the document and chunk records

Returns:
- `document_id`
- `title`
- `filename`
- `chunks_inserted`

---

### `POST /retrieve`
Runs the default retrieval pipeline.

What it does:
1. embeds the query
2. retrieves top-k chunks by vector similarity
3. returns ranked chunk results with metadata

Use this when:
- you want raw vector retrieval results
- you want to inspect baseline retrieval behavior

---

### `POST /retrieve_rerank`
Runs retrieval plus lightweight post-retrieval reranking.

What it does:
1. performs the same vector retrieval as `/retrieve`
2. applies lightweight query normalization
3. adds lexical overlap / weighted keyword bonuses
4. reorders the retrieved candidates

Important:
- reranking changes the **order** of the results
- it does **not** change the original cosine distances

Use this when:
- you want to inspect improved retrieval ordering
- you want to compare baseline retrieval vs reranked retrieval

---

### `POST /ask`
Runs the default grounded QA pipeline.

What it does:
1. embeds the query
2. retrieves raw candidates
3. applies refusal and similarity thresholds
4. selects evidence
5. generates a grounded answer
6. extracts cited chunk IDs from the answer
7. returns only the chunks actually cited

Use this when:
- you want the current default QA behavior

Notes:
- refusal logic is based on retrieval similarity thresholds
- conclusion-style questions may trigger an intent gate
- this is currently the recommended default QA endpoint

---

### `POST /ask_rerank`
Runs grounded QA with reranked evidence ordering.

What it does:
1. embeds the query
2. retrieves raw candidates
3. keeps refusal logic based on the original retrieval distances
4. reranks the retrieved rows
5. selects reranked evidence for answer generation
6. generates a grounded answer with citations

Use this when:
- you want to experimentally test whether improved retrieval ordering helps downstream QA

Current status:
- useful for experiments
- not currently recommended as the default QA path

---

### `GET /documents`
Lists ingested documents for debugging or UI selection.

Useful for:
- checking available `document_id` values
- confirming ingestion status

---

### `GET /documents/{document_id}`
Returns details for a specific ingested document.

Useful for:
- inspecting stored document metadata
- debugging document-level state

---

### `GET /health`
Simple service health check.

Useful for:
- confirming that the API server is running

---

## Current Recommended Defaults

- Default QA path: `/ask`
- Experimental retrieval improvement: `/retrieve_rerank`
- Experimental QA path: `/ask_rerank`

---

## Quick Start

### 1. Start the database
The project includes infrastructure under `infra/`.

If you are using Docker Compose:

```bash
docker compose -f infra/docker-compose.yml up -d
```

If your environment uses the older command:

```bash
docker-compose -f infra/docker-compose.yml up -d
```

---

### 2. Set environment variables
At minimum, the app needs a valid database connection and embedding/generation configuration.

Typical example:

```bash
export DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/DBNAME
```

You may also need to export your model API key, depending on how `embeddings.py` and generation are configured.

---

### 3. Start the API server

```bash
uvicorn backend.app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

to inspect the API interactively.

---

## Example Usage

### Ingest a markdown file

```bash
curl -sS -X POST "http://127.0.0.1:8000/ingest" \
  -F "file=@eval/fixtures/docs/test_eval_notes.md" \
  -F "title=Test Eval Notes Markdown" \
  -F "replace=true"
```

### Retrieve evidence

```bash
curl -sS -X POST "http://127.0.0.1:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What is the purpose of the project?",
    "top_k": 4,
    "document_id": 4
  }'
```

### Retrieve with reranking

```bash
curl -sS -X POST "http://127.0.0.1:8000/retrieve_rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What is the purpose of the project?",
    "top_k": 4,
    "document_id": 4
  }'
```

### Ask a grounded question

```bash
curl -sS -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What is the purpose of the project?",
    "top_k": 4,
    "document_id": 4
  }'
```

---

## Evaluation Workflow

This project includes a small but structured evaluation workflow.

### Baseline validation
The initial baseline established that:
- markdown ingestion works
- PDF ingestion works
- retrieval works
- grounded QA works
- citation extraction works
- refusal behavior works

### Retrieval analysis
Retrieval-only analysis showed:
- gold evidence reliably entered top-k
- the main weakness was **ranking rather than recall**
- purpose/objective-style paraphrase questions were more likely to miss at top1

### Retrieve vs Retrieve-Rerank A/B
A focused A/B comparison between `/retrieve` and `/retrieve_rerank` showed:

- retrieve top-1 hit rate: **2/7**
- retrieve_rerank top-1 hit rate: **5/7**
- repaired cases: **3**
- worsened cases: **0**

This showed that lightweight query-normalized lexical reranking substantially improved retrieval ordering without harming already-correct top-1 cases in the tested subset.

### Ask vs Ask-Rerank A/B
A downstream A/B comparison between `/ask` and `/ask_rerank` showed:

- ask gold-citation hits: **5/5**
- ask_rerank gold-citation hits: **5/5**
- improved cases: **0**
- worsened cases: **0**

Interpretation:
the original `/ask` pipeline was already strong at selecting the correct evidence from the retrieved top-k candidates. Therefore, while reranking improved retrieval ordering, it did not improve final QA metrics on the current small ask-level test set.

---

## Current Interpretation

At the current stage:

- the system already performs well on small grounded QA tests
- retrieval recall is strong
- retrieval ranking is the main weak point
- lightweight reranking improves ranking quality
- downstream QA is currently robust enough that reranking has not yet improved small ask-level metrics

So the reranker is best understood as a **retrieval-layer enhancement** rather than a proven default QA improvement.

---

## Limitations

Current limitations include:

- no OCR path for scanned PDFs
- no learned cross-encoder reranker
- limited test set size
- current rerank is heuristic and query-normalized, not learned
- downstream QA evaluation is still relatively small
- PDF layout parsing is useful but still heuristic rather than layout-model-based

---

## Future Work

Potential next steps include:

- add OCR support for scanned PDFs
- expand evaluation sets with harder and more adversarial queries
- test retrieval improvements on larger question sets
- add a learned reranker for comparison
- experiment with hybrid retrieval
- improve PDF layout parsing beyond simple two-column heuristics
- expand answer-level evaluation beyond citation hit rate

---

## Practical Takeaway

If you are using this project today:

- use `/ask` as the default QA endpoint
- use `/retrieve_rerank` when you want improved retrieval ordering for experiments
- keep `/ask_rerank` as a research/testing path rather than the default

This project is intended to stay small, inspectable, and useful for understanding how retrieval, grounding, reranking, and refusal interact in a real RAG pipeline.

### Phase 1 expanded retrieval evaluation

We added 10 additional evaluation questions under the `phase1_expand` subset, focusing on `purpose_objective` and `numeric_boundary` questions. These questions use gold chunks from `test_eval_notes.md` and `test_dual_column_eval.pdf`.

On this subset, the baseline retrieval endpoint achieved 9/10 top-1 gold hits, while the lightweight reranked retrieval endpoint achieved 10/10 top-1 gold hits. Reranking repaired one `purpose_objective` case and introduced no regressions.

This suggests that the reranker is safe on this subset and can improve ranking for semantically ambiguous purpose/workflow questions, although the current subset is still relatively easy and should be expanded with harder numeric-boundary and purpose-objective cases.

### Numeric-boundary reranking update

We added a small numeric-boundary reranking bonus to improve evidence ranking for questions involving protocol duration and nearby numeric facts. The failure case was not a recall failure: the correct chunk was already retrieved but ranked lower than conclusion/result chunks.

Before this update, the hard numeric-boundary subset achieved 2/4 top-1 hits for both baseline retrieval and reranked retrieval. After adding duration-aware lexical boosts for terms such as `minutes`, `protocol`, `exposed`, and `recover`, reranked retrieval improved to 4/4 top-1 hits on the hard subset, with no regressions on the 10-question `phase1_expand` sanity set.

This suggests that lightweight reranking can repair specific evidence-localization failures when the gold evidence is already present in the retrieved candidate set.