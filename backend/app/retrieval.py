from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

from .db import get_conn


def retrieve_top_k(query_vec: np.ndarray, top_k: int, document_id: int | None = None):
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


def citations_from_rows(rows: list[tuple]) -> list[Dict[str, Any]]:
    """
    rows schema after JOIN:
      [(cid, chunk_index, content, metadata, dist, title, filename), ...]
    but we parse defensively in case columns change later.
    """
    out: list[Dict[str, Any]] = []
    for row in rows:
        # stable core fields
        cid, chunk_index, content, metadata, dist = row[:5]

        # optional doc fields from JOIN
        title = row[5] if len(row) > 5 else None
        filename = row[6] if len(row) > 6 else None

        if isinstance(metadata, dict):
            page = metadata.get("page")
            section = metadata.get("section")
            level = metadata.get("level")
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
                "filename_meta": filename_meta,
                "source_type": source_type,
                "page": page,
                "section": section,
                "level": level,
                "cosine_distance": float(dist),
                "snippet": content[:250],
            }
        )
    return out


# ---------------------------------------------------------------------
# Lightweight rerank
# ---------------------------------------------------------------------

BOOST_TERMS: dict[str, float] = {
    # purpose / objective / goal
    "purpose": 0.25,
    "objective": 0.25,
    "goal": 0.20,
    "main": 0.08,
    "directly": 0.06,
    "exact": 0.06,

    # experiment / study framing
    "experimental": 0.15,
    "study": 0.10,
    "built": 0.05,
    "around": 0.03,

    # method terms
    "measured": 0.15,
    "assay": 0.15,
    "method": 0.12,
    "approach": 0.10,

    # limitations / negative evidence / benchmarks
    "benchmark": 0.15,
    "accuracy": 0.10,
    "report": 0.10,
    "reported": 0.10,
    "limitation": 0.12,
    "constraint": 0.10,
    "current": 0.06,
    "not": 0.05,
    "no": 0.05,

    # architecture / metadata
    "section": 0.08,
    "metadata": 0.12,
    "cite": 0.08,
    "citation": 0.08,
    # numeric boundary / protocol duration
    "duration": 0.18,
    "minute": 0.16,
    "minutes": 0.16,
    "protocol": 0.14,
    "exposed": 0.12,
    "recover": 0.12,
    "recovery": 0.10,
    "phase": 0.08,
    "peak": 0.10,
    "value": 0.08,
}


def normalize_query(text: str) -> str:
    t = text.lower()

    # abstract intent normalization
    t = t.replace("experimental goal", "objective")
    t = t.replace("built around", "objective")
    t = t.replace("main purpose", "purpose")
    t = t.replace("main objective", "objective")

    # weaker paraphrase normalization
    t = t.replace("current constraint", "limitation")
    t = t.replace("current constraints", "limitations")
    t = t.replace("what limitation", "limitation")
    t = t.replace("measurement approach", "method assay")
    t = t.replace("helpful for evaluation", "easy to retrieve evaluation")
    t = t.replace("baseline value", "baseline numeric value")
    t = t.replace("before stress began", "baseline before stress began")

    return t


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def numeric_boundary_bonus(query: str, text: str) -> float:
    q = normalize_query(query)
    q_tokens = set(tokenize(q))
    t_tokens = set(tokenize(text))

    bonus = 0.0

    asks_duration = (
        "duration" in q_tokens
        or ("how" in q_tokens and "long" in q_tokens)
        or "minutes" in q_tokens
        or "minute" in q_tokens
    )

    if asks_duration:
        # Strongly reward chunks that actually contain timing/protocol evidence.
        if "minutes" in t_tokens or "minute" in t_tokens:
            bonus += 0.35
        if "protocol" in t_tokens:
            bonus += 0.20
        if "exposed" in t_tokens:
            bonus += 0.15
        if "recover" in t_tokens or "recovery" in t_tokens:
            bonus += 0.12

        # Penalize pure conclusion/result chunks for duration questions.
        if "conclusion" in t_tokens:
            bonus -= 0.12
        if "result" in t_tokens or "results" in t_tokens:
            bonus -= 0.08

    asks_numeric_value = (
        "value" in q_tokens
        or "peak" in q_tokens
        or "mean" in q_tokens
        or "units" in q_tokens
    )

    if asks_numeric_value:
        if "value" in t_tokens or "mean" in t_tokens:
            bonus += 0.18
        if "units" in t_tokens:
            bonus += 0.12
        if "peak" in t_tokens:
            bonus += 0.12

    return bonus


def lexical_bonus(query: str, text: str) -> float:
    q_norm = normalize_query(query)
    q_tokens = set(tokenize(q_norm))
    t_tokens = set(tokenize(text))

    bonus = 0.0

    # overlap bonus
    overlap = q_tokens & t_tokens
    bonus += 0.03 * len(overlap)

    # stronger term-weight bonus
    for tok, weight in BOOST_TERMS.items():
        if tok in q_tokens and tok in t_tokens:
            bonus += weight

    bonus += numeric_boundary_bonus(query, text)
    return bonus


def rerank_score(query: str, result: dict[str, Any]) -> float:
    """
    Higher is better.

    score = -cosine_distance + lexical_bonus(query, snippet)
    """
    dist = float(result.get("cosine_distance", 999.0))
    snippet = str(result.get("snippet", ""))
    return (-dist) + lexical_bonus(query, snippet)


def rerank_results(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []
    for r in results:
        x = dict(r)
        x["rerank_score"] = rerank_score(query, r)
        reranked.append(x)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


def rerank_rows(query: str, rows: list[tuple]) -> list[tuple]:
    """
    Rerank raw DB rows by:
      1) converting rows -> citation-style dicts
      2) applying rerank_results(query, ...)
      3) mapping back to the original rows

    Important:
      - this changes ORDER only
      - it does NOT recompute cosine distances
      - threshold logic should still use the original raw rows
    """
    if not rows:
        return rows

    retrieved = citations_from_rows(rows)
    reranked = rerank_results(query, retrieved)

    row_by_chunk_id = {row[0]: row for row in rows}  # row[0] = chunk_id
    out: list[tuple] = []

    for r in reranked:
        cid = r.get("chunk_id")
        if cid in row_by_chunk_id:
            out.append(row_by_chunk_id[cid])

    return out
