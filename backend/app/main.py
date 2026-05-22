from fastapi import HTTPException
from .embeddings import embed_queries
from .llm import generate_answer
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .embeddings import embed_documents, embed_queries
from psycopg.types.json import Jsonb
import re

from fastapi import UploadFile, File, Form, HTTPException
from .parsers import parse_pdf, parse_md
from .chunking import chunk_units
from .store import sha256_bytes, upsert_document, insert_chunks
from .embeddings import embed_documents
from fastapi import Query
from typing import Any, Dict, List
import time
from hashlib import sha256
import os
from dotenv import load_dotenv
from .retrieval import retrieve_top_k, citations_from_rows, rerank_results


load_dotenv()

FILTER_MAX_COSINE_DISTANCE = float(
    os.getenv("RAG_FILTER_MAX_COSINE_DISTANCE", "0.85"))
REFUSE_IF_BEST_DISTANCE_GT = float(
    os.getenv("RAG_REFUSE_IF_BEST_DISTANCE_GT", "0.75"))

# 全局内存缓存：key -> (时间戳, 响应字典)
_ANSWER_CACHE = {}
# 缓存有效期：300 秒（5 分钟），对于科研调试足够了
CACHE_TTL_SECONDS = 300

CONCLUSION_TERMS = (
    "conclusion",
    "conclusions",
    "in conclusion",
    "we conclude",
    "overall",
    "summary",
    "takeaway",
)


def _needs_conclusion_evidence(q: str) -> bool:
    ql = (q or "").lower()
    return "final conclusion" in ql or "conclusion" in ql


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    tl = (text or "").lower()
    return any(t in tl for t in terms)


app = FastAPI(title="RAG Doc QA")
# 0.45 是一个经验值，0 表示完全相同，1 表示完全无关
# 在 Demo 阶段，这个值能有效过滤掉跨度太大的内容


@app.get("/health")
def health():
    return {"ok": True}


class SearchTextRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(2, ge=1, le=20)


class AskRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=20)
    document_id: int | None = None  # 默认为 None，表示搜索全库；传入数字则只搜该文档


@app.post("/seed_demo")
def seed_demo():
    """
    Resets the vector database and populates it with mock RAG chunks using real Gemini embeddings.

    This sandbox routing endpoint wipes all historical data from both the `chunks` 
    and `documents` tables and resets the auto-incrementing primary key sequences. 
    It then registers a dummy parent document named "demo-doc", batch-embeds 3 static 
    sample text chunks into 768-dimensional dense vectors via the Gemini API, and 
    inserts them with associated metadata (page numbers). 

    This is primarily used to provide a clean, isolated, and deterministic playground 
    for validation testing of vector similarity searches and downstream QA pipelines.

    Returns:
        dict: A JSON response containing a success flag, the newly generated 
              parent `document_id`, and the total number of `chunks_inserted`.
    """

    chunks = [
        ("Cats are small domesticated animals.", {"page": 1}),
        ("Dogs are loyal and often used as pets.", {"page": 2}),
        ("SQL databases store structured data.", {"page": 3}),
    ]

    texts = [c[0] for c in chunks]
    vectors = embed_documents(texts)  # list[np.ndarray], each 768-d

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE TABLE documents RESTART IDENTITY CASCADE;")

            cur.execute(
                "INSERT INTO documents (title, source) VALUES (%s, %s) RETURNING id;",
                ("demo-doc", "seed_demo_gemini_embeddings"),
            )
            doc_id = cur.fetchone()[0]

            for idx, ((text, meta), vec) in enumerate(zip(chunks, vectors)):
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s::jsonb, %s);
                    """,
                    (doc_id, idx, text, Jsonb(meta), vec),
                )

    return {"ok": True, "document_id": doc_id, "chunks_inserted": len(chunks)}


@app.post("/search_text")
def search_text(req: SearchTextRequest) -> Dict[str, Any]:
    """
    Performs a raw vector similarity search against all stored chunks.

    This routing endpoint takes a user's plain text query, transforms it into a 
    single 768-dimensional dense vector via the query embedding pipeline, and 
    executes a high-dimensional geometric search across the `chunks` table using 
    the pgvector cosmic operator (`<=>`). It computes both the cosine distance 
    and cosine similarity, orders the matches by proximity, and enforces a strict 
    upper bound via `top_k`.

    Args:
        req (SearchTextRequest): A validated Pydantic model containing the 
            `query_text` string and an integer `top_k` (bounded between 1 and 20).

    Returns:
        dict: A serialized JSON response containing the original metadata, 
              the requested `top_k`, and a structured `results` list mapping 
              individual hit parameters (distance, similarity, content, etc.).
    """
    q_vec = embed_queries([req.query_text])[0]  # single 768-d vector

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, chunk_index, content, metadata,
                    (embedding <=> %s) AS cosine_distance,
                    (1 - (embedding <=> %s)) AS cosine_similarity
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (q_vec, q_vec, q_vec, req.top_k),
            )
            rows = cur.fetchall()

    results = []
    for (cid, chunk_index, content, metadata, dist, sim) in rows:
        results.append(
            {
                "chunk_id": cid,
                "chunk_index": chunk_index,
                "content": content,
                "metadata": metadata,
                "cosine_distance": float(dist),   # 越小越好
                "cosine_similarity": float(sim),  # 越大越好
            }
        )

    return {"ok": True, "query_text": req.query_text, "top_k": req.top_k, "results": results}


def retrieve_top_k(query_vec: np.ndarray, top_k: int, document_id: int | None = None):
    """
    Retrieves the nearest database chunks with parent document metadata.

    This low-level core RAG utility interacts directly with PostgreSQL to fetch 
    the most semantically relevant text fragments. It implements two operational modes:
    1. Global Search (when `document_id` is None): Scans the entire vector database.
    2. Targeted Search (when `document_id` is provided): Restricts the computation 
       scope to chunks belonging to a specific document via an optimized `WHERE` clause.

    Both modes utilize an inner `JOIN` operation on the `documents` table to 
    dynamically pull the parent document's `title` and `filename`, serving as the 
    foundational ledger data for upstream LLM response grounding and citations.

    Args:
        query_vec (np.ndarray / list): The 768-dimensional embedding vector of the query.
        top_k (int): Maximum number of records to return from the sorted min-heap.
        document_id (int | None, optional): Primary key of the targeted document 
            for scoped retrieval. Defaults to None (global cross-document search).

    Returns:
        list of tuples: A database cursor record list where each row contains stable 
                        chunk fragments alongside joined parent document metadata fields:
                        (id, chunk_index, content, metadata, cosine_distance, title, filename).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            if document_id is None:
                cur.execute(
                    """
                    SELECT
                      c.id, c.chunk_index, c.content, c.metadata,
                      (c.embedding <=> %s) AS cosine_distance,
                      d.title, d.filename
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    ORDER BY c.embedding <=> %s
                    LIMIT %s;
                    """,
                    (query_vec, query_vec, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT
                      c.id, c.chunk_index, c.content, c.metadata,
                      (c.embedding <=> %s) AS cosine_distance,
                      d.title, d.filename
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.document_id = %s
                    ORDER BY c.embedding <=> %s
                    LIMIT %s;
                    """,
                    (query_vec, document_id, query_vec, top_k),
                )
            return cur.fetchall()


def extract_cited_chunk_ids(answer: str) -> set[int]:
    """
    Extract cited ids from patterns like: [1], [12], [1,2], [1, 2]
    """
    ids: set[int] = set()
    for group in re.findall(r"\[([0-9,\s]+)\]", answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
    return ids


def citations_from_rows(rows: list[tuple]) -> list[Dict[str, Any]]:
    """
    Parses an LLM-generated answer text to extract all unique cited chunk IDs.

    This utility functions as a post-processing step in the RAG pipeline. It utilizes 
    a localized regular expression pattern `r"\[([0-9,\s]+)\]"` to capture standard 
    academic citation brackets like `[1]`, `[12]`, or multi-citations like `[1, 2]`. 
    Captured groups are tokenized, stripped of whitespaces, and deduplicated via a 
    mathematical integer set to prepare clean alignment mappings for frontend UI 
    hyperlink rendering.

    Args:
        answer (str): The raw text response string returned by the downstream LLM 
            containing potential inline bracketed citations.

    Returns:
        set[int]: A deduplicated set of integers representing the precise database 
                  `chunk_id`s explicitly referenced in the response text.
    """
    out: list[Dict[str, Any]] = []
    for row in rows:
        # always take the first 5 (stable)
        cid, chunk_index, content, metadata, dist = row[:5]

        # optional doc fields from JOIN
        title = row[5] if len(row) > 5 else None
        filename = row[6] if len(row) > 6 else None

    # --- 这里是 Step 8.2-5 新增的提取逻辑 ---
        if isinstance(metadata, dict):
            page = metadata.get("page")
            section = metadata.get("section")
            level = metadata.get("level")
            # 新增：从 meta 字典里拿 source_type 和 filename
            source_type = metadata.get("source_type")
            filename_meta = metadata.get("filename")
        else:
            page = section = level = source_type = filename_meta = None

        out.append(
            {
                "chunk_id": cid,
                "chunk_index": chunk_index,
                "title": title,
                "filename": filename,
                "filename_meta": filename_meta,  # 这是从 chunk 冗余字段拿的
                "source_type": source_type,     # 新增
                "page": page,
                "section": section,
                "level": level,
                "cosine_distance": float(dist),
                "snippet": content[:250],
            }
        )
    return out


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    replace: bool = Form(True),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = file.filename or "uploaded"
    mime_type = file.content_type or "application/octet-stream"
    digest = sha256_bytes(data)

    # 1) parse
    lower = filename.lower()
    if lower.endswith(".pdf"):
        units = parse_pdf(data)
        source = "pdf"
    elif lower.endswith(".md") or lower.endswith(".markdown") or lower.endswith(".txt"):
        units = parse_md(data)
        source = "md"
    else:
        raise HTTPException(
            status_code=400, detail="Only .pdf / .md / .txt supported for now")

    if not units:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found. If it's a scanned PDF, you'll need OCR (we can add later).",
        )

    # 2) chunk
    chunks = chunk_units(units, max_words=250, overlap_words=50)
    if not chunks:
        raise HTTPException(
            status_code=400, detail="Chunking produced 0 chunks")

    # 2.5) enrich metadata for every chunk (research-friendly)
    for c in chunks:
        meta = c.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        meta["source_type"] = source   # "pdf" or "md"
        meta["filename"] = filename
        c["meta"] = meta

    # 3) embed (document task)
    texts = [c["text"] for c in chunks]

    # batching to avoid too-large requests
    vectors = []
    BATCH = 32
    for i in range(0, len(texts), BATCH):
        vectors.extend(embed_documents(texts[i: i + BATCH]))

    # 4) store
    doc_title = title or filename
    doc_id = upsert_document(
        title=doc_title,
        source=source,
        filename=filename,
        mime_type=mime_type,
        sha256=digest,
        replace=replace,
    )
    n = insert_chunks(doc_id, chunks, vectors)

    return {
        "ok": True,
        "document_id": doc_id,
        "title": doc_title,
        "filename": filename,
        "mime_type": mime_type,
        "sha256": digest,
        "chunks_inserted": n,
    }


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    """
    RAG endpoint:
      1) embed query
      2) retrieve raw_k candidates (cosine distance)
      3) threshold filter -> selected (<= MAX_COSINE_DISTANCE)
      4) if none -> refuse (no LLM call)
      5) generate grounded answer with citations like [chunk_id]
      6) return only citations that were actually cited (fallback to top-1 if none)
    """
    # --- 13.7 新增：构造缓存 Key (基于文档ID、问题内容和 top_k) ---
    cache_raw_str = f"{req.document_id}|{req.query_text}|{req.top_k}"
    cache_key = sha256(cache_raw_str.encode()).hexdigest()

    now = time.time()
    if cache_key in _ANSWER_CACHE:
        ts, cached_response = _ANSWER_CACHE[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            # 命中缓存！直接返回，不消耗 Gemini 配额
            return cached_response
    # --- 缓存未命中，继续执行原有逻辑 ---

    # 1) query embedding
    q_vec = embed_queries([req.query_text])[0]

    # 2) retrieve more than top_k, then filter
    raw_k = min(req.top_k * 3, 20)  # cap
    # [(cid, idx, content, metadata, cosine_distance), ...]
    rows_raw = retrieve_top_k(q_vec, raw_k, document_id=req.document_id)

    best_dist = float(rows_raw[0][4]) if rows_raw else None

# 证据最强的一条都不够相似 → 直接拒答（不调用 LLM）
    if best_dist is None or best_dist > REFUSE_IF_BEST_DISTANCE_GT:
        return {
            "ok": True,
            "query_text": req.query_text,
            "answer": "I don't know based on the provided documents.",
            "citations": [],
            "retrieved_chunks": citations_from_rows(rows_raw),
            "selected_chunks": [],
            "threshold": {
                "filter_max_cosine_distance": FILTER_MAX_COSINE_DISTANCE,
                "refuse_if_best_distance_gt": REFUSE_IF_BEST_DISTANCE_GT,
                "best_cosine_distance": best_dist,
                "raw_k": raw_k,
                "top_k": req.top_k,
                "kept_after_filter": 0,
            },
            "cited_chunk_ids": [],
            "refused": True,
            "reason": "Best chunk not similar enough.",
        }

    # 3) similarity threshold filter
    rows_filtered = [r for r in rows_raw if float(
        r[4]) <= FILTER_MAX_COSINE_DISTANCE]
    rows_selected = rows_filtered[: req.top_k]

    retrieved_chunks = citations_from_rows(rows_raw)
    selected_chunks = citations_from_rows(rows_selected)

    # 4) refuse if no sources passed threshold
    if len(rows_selected) == 0:
        return {
            "ok": True,
            "query_text": req.query_text,
            "answer": "I don't know based on the provided documents.",
            "citations": [],
            "retrieved_chunks": retrieved_chunks,
            "selected_chunks": [],
            "threshold": {
                "filter_max_cosine_distance": FILTER_MAX_COSINE_DISTANCE,
                "refuse_if_best_distance_gt": REFUSE_IF_BEST_DISTANCE_GT,
                "best_cosine_distance": float(rows_raw[0][4]) if rows_raw else None,
                "raw_k": raw_k,
                "top_k": req.top_k,
                "kept_after_filter": len(rows_filtered),
            },
            "cited_chunk_ids": [],
            "refused": True,
            "reason": "No chunks passed similarity threshold.",
        }

    # 5) prepare sources for LLM (use ONLY selected rows)
    sources_for_llm: list[Dict[str, Any]] = []
    for row in rows_selected:
        cid, chunk_index, content, metadata, dist = row[:5]
        page = metadata.get("page") if isinstance(metadata, dict) else None
        sources_for_llm.append(
            {"chunk_id": cid, "page": page, "content": content}
        )

    # ---- Intent gate: require conclusion-like evidence when asked for conclusion ----
    if _needs_conclusion_evidence(req.query_text):
        joined = "\n".join([s["content"] for s in sources_for_llm])
        if not _has_any_term(joined, CONCLUSION_TERMS):
            return {
                "ok": True,
                "query_text": req.query_text,
                "answer": "I don't know based on the provided documents.",
                "citations": [],
                "retrieved_chunks": citations_from_rows(rows_raw),
                "selected_chunks": citations_from_rows(rows_selected),
                "threshold": {
                    "filter_max_cosine_distance": FILTER_MAX_COSINE_DISTANCE,
                    "refuse_if_best_distance_gt": REFUSE_IF_BEST_DISTANCE_GT,
                    "best_cosine_distance": best_dist,
                    "raw_k": raw_k,
                    "top_k": req.top_k,
                    "kept_after_filter": len(rows_filtered),
                    "intent_gate": "conclusion_terms_missing",
                },
                "cited_chunk_ids": [],
                "refused": True,
                "reason": "Conclusion requested but no conclusion-like evidence in retrieved sources.",
            }

    answer = generate_answer(req.query_text, sources_for_llm)

    # 6) citation filtering: keep only actually cited chunks
    used_ids = extract_cited_chunk_ids(answer)
    if used_ids:
        citations_used_only = [
            c for c in selected_chunks if c["chunk_id"] in used_ids]
    else:
        # fallback: at least return top-1 selected source
        citations_used_only = selected_chunks[:1]
        if citations_used_only:
            used_ids = {citations_used_only[0]["chunk_id"]}

    citation_text_lines = []
    for c in citations_used_only:
        fn = c.get("filename") or c.get("filename_meta") or "unknown"
        pg = c.get("page")
        chunk_id = c.get("chunk_id")
        snip = c.get("snippet", "").replace("\n", " ").strip()
        citation_text_lines.append(
            f"{fn} p.{pg} (chunk {chunk_id}): {snip}"
            if pg is not None else
            f"{fn} (chunk {chunk_id}): {snip}"
        )
    citation_text = "\n".join(citation_text_lines)

    final_response = {
        "ok": True,
        "query_text": req.query_text,
        "answer": answer,
        "citations": citations_used_only,
        "citation_text": citation_text,
        "cited_chunk_ids": sorted(list(used_ids)),
        "refused": False,
        # 可以加个标志位，方便你在 UI 调试时知道这是不是缓存
        "from_cache": False,
        "threshold": {
            "filter_max_cosine_distance": FILTER_MAX_COSINE_DISTANCE,
            "refuse_if_best_distance_gt": REFUSE_IF_BEST_DISTANCE_GT,
            "best_cosine_distance": best_dist,
            "raw_k": raw_k,
            "top_k": req.top_k,
            "kept_after_filter": len(rows_filtered),
        }
    }

    # --- 13.7 新增：写入缓存 ---
    _ANSWER_CACHE[cache_key] = (
        time.time(), {**final_response, "from_cache": True})

    return final_response


@app.get("/documents")
def list_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    List ingested documents for UI dropdown / debugging.
    Returns newest first.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, filename, mime_type, sha256, created_at
                FROM documents
                ORDER BY created_at DESC, id DESC
                LIMIT %s OFFSET %s;
                """,
                (limit, offset),
            )
            rows = cur.fetchall()

    items: List[Dict[str, Any]] = []
    for (doc_id, title, filename, mime_type, sha256, created_at) in rows:
        items.append(
            {
                "id": doc_id,
                "title": title,
                "filename": filename,
                "mime_type": mime_type,
                "sha256": sha256,
                "created_at": created_at.isoformat() if created_at else None,
            }
        )

    return {"ok": True, "count": len(items), "items": items, "limit": limit, "offset": offset}


@app.get("/documents/{document_id}")
def get_document(document_id: int) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, filename, mime_type, sha256, created_at
                FROM documents
                WHERE id = %s;
                """,
                (document_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_id, title, filename, mime_type, sha256, created_at = row
    return {
        "ok": True,
        "item": {
            "id": doc_id,
            "title": title,
            "filename": filename,
            "mime_type": mime_type,
            "sha256": sha256,
            "created_at": created_at.isoformat() if created_at else None,
        },
    }


class RetrieveRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=20)
    document_id: int | None = None


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    q_vec = embed_queries([req.query_text])[0]
    rows = retrieve_top_k(q_vec, req.top_k, document_id=req.document_id)
    retrieved = citations_from_rows(rows)
    return {"ok": True, "query_text": req.query_text, "top_k": req.top_k, "results": retrieved}
