from .embeddings import embed_queries
from .llm import generate_answer
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict

from .db import get_conn
from .embeddings import embed_documents, embed_queries
from psycopg.types.json import Jsonb
import re

from fastapi import UploadFile, File, Form, HTTPException
from .parsers import parse_pdf, parse_md
from .chunking import chunk_units
from .store import sha256_bytes, upsert_document, insert_chunks
from .embeddings import embed_documents

app = FastAPI(title="RAG Doc QA")
# 0.45 是一个经验值，0 表示完全相同，1 表示完全无关
# 在 Demo 阶段，这个值能有效过滤掉跨度太大的内容
MAX_COSINE_DISTANCE = 0.45


@app.get("/health")
def health():
    return {"ok": True}


class SearchTextRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(2, ge=1, le=20)


class AskRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=20)


@app.post("/seed_demo")
def seed_demo():
    """
    Seed 3 chunks with REAL Gemini embeddings (768-d).
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
    # 这里的变量要和 SELECT 里的列一一对应
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


def retrieve_top_k(query_vec, top_k: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, chunk_index, content, metadata,
                       (embedding <=> %s) AS cosine_distance
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (query_vec, query_vec, top_k),
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
    rows schema:
      [(cid, chunk_index, content, metadata, cosine_distance), ...]
    """
    out: list[Dict[str, Any]] = []
    for (cid, chunk_index, content, metadata, dist) in rows:
        page = metadata.get("page") if isinstance(metadata, dict) else None
        out.append(
            {
                "chunk_id": cid,
                "chunk_index": chunk_index,
                "page": page,
                "cosine_distance": float(dist),
                "snippet": content[:250],
            }
        )
    return out


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
    # 1) query embedding
    q_vec = embed_queries([req.query_text])[0]

    # 2) retrieve more than top_k, then filter
    raw_k = min(req.top_k * 3, 20)  # cap
    # [(cid, idx, content, metadata, cosine_distance), ...]
    rows_raw = retrieve_top_k(q_vec, raw_k)

    # 3) similarity threshold filter
    rows_filtered = [r for r in rows_raw if float(
        r[-1]) <= MAX_COSINE_DISTANCE]
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
                "max_cosine_distance": MAX_COSINE_DISTANCE,
                "raw_k": raw_k,
                "top_k": req.top_k,
                "kept_after_threshold": 0,
            },
            "cited_chunk_ids": [],
            "refused": True,
            "reason": "No chunks passed similarity threshold.",
        }

    # 5) prepare sources for LLM (use ONLY selected rows)
    sources_for_llm: list[Dict[str, Any]] = []
    for (cid, chunk_index, content, metadata, dist) in rows_selected:
        page = metadata.get("page") if isinstance(metadata, dict) else None
        sources_for_llm.append(
            {
                "chunk_id": cid,
                "page": page,
                "content": content,
            }
        )

    answer = generate_answer(req.query_text, sources_for_llm)

    # 6) citation filtering: keep only actually cited chunks
    used_ids = extract_cited_chunk_ids(answer)
    if used_ids:
        citations_used_only = [
            c for c in selected_chunks if c["chunk_id"] in used_ids]
    else:
        # fallback: at least return top-1 selected source
        citations_used_only = selected_chunks[:1]

    return {
        "ok": True,
        "query_text": req.query_text,
        "answer": answer,
        "citations": citations_used_only,
        "retrieved_chunks": retrieved_chunks,   # raw candidates (debug/eval)
        # after threshold + top_k (debug)
        "selected_chunks": selected_chunks,
        "threshold": {
            "max_cosine_distance": MAX_COSINE_DISTANCE,
            "raw_k": raw_k,
            "top_k": req.top_k,
            "kept_after_threshold": len(rows_filtered),
        },
        "cited_chunk_ids": sorted(list(used_ids)),
        "refused": False,
    }


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
