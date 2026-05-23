from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
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
# Lightweight configurable rerank
# ---------------------------------------------------------------------

PROFILE_PATH = Path(__file__).with_name("rerank_profiles.json")


FALLBACK_PROFILE: dict[str, Any] = {
    "boost_terms": {
        "purpose": 0.25,
        "objective": 0.25,
        "goal": 0.20,
        "method": 0.12,
        "assay": 0.15,
        "limitation": 0.12,
        "metadata": 0.12,
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
    },
    "query_normalizations": {
        "experimental goal": "objective",
        "built around": "objective",
        "main purpose": "purpose",
        "main objective": "objective",
        "current constraint": "limitation",
        "current constraints": "limitations",
        "what limitation": "limitation",
        "measurement approach": "method assay",
        "helpful for evaluation": "easy to retrieve evaluation",
        "baseline value": "baseline numeric value",
        "before stress began": "baseline before stress began",
    },
    "numeric_boundary": {
        "duration_query_terms": ["duration", "how long", "minutes", "minute"],
        "duration_positive_terms": {
            "minutes": 0.35,
            "minute": 0.35,
            "protocol": 0.20,
            "exposed": 0.15,
            "recover": 0.12,
            "recovery": 0.12,
        },
        "duration_negative_terms": {
            "conclusion": -0.12,
            "result": -0.08,
            "results": -0.08,
        },
        "value_query_terms": ["value", "peak", "mean", "units"],
        "value_positive_terms": {
            "value": 0.18,
            "mean": 0.18,
            "units": 0.12,
            "peak": 0.12,
        },
    },
}


@lru_cache(maxsize=1)
def load_rerank_profiles() -> dict[str, Any]:
    """
    Load rerank profiles from JSON.

    The profile file allows reranking behavior to be tuned without editing
    Python code. If the file is missing or malformed, fall back to a safe
    built-in default profile so retrieval endpoints remain usable.
    """
    if not PROFILE_PATH.exists():
        return {"default": FALLBACK_PROFILE}

    try:
        with PROFILE_PATH.open("r", encoding="utf-8") as f:
            profiles = json.load(f)
    except Exception:
        return {"default": FALLBACK_PROFILE}

    if not isinstance(profiles, dict):
        return {"default": FALLBACK_PROFILE}

    if "default" not in profiles or not isinstance(profiles["default"], dict):
        profiles["default"] = FALLBACK_PROFILE

    return profiles


def get_rerank_profile(profile_name: str = "default") -> dict[str, Any]:
    profiles = load_rerank_profiles()
    profile = profiles.get(profile_name)

    if not isinstance(profile, dict):
        return profiles["default"]

    return profile


def get_weight_map(profile: dict[str, Any], key: str) -> dict[str, float]:
    raw = profile.get(key, {})
    if not isinstance(raw, dict):
        return {}

    out: dict[str, float] = {}
    for term, weight in raw.items():
        try:
            out[str(term).lower()] = float(weight)
        except (TypeError, ValueError):
            continue

    return out


def get_nested_weight_map(profile: dict[str, Any], section: str, key: str) -> dict[str, float]:
    raw_section = profile.get(section, {})
    if not isinstance(raw_section, dict):
        return {}

    raw = raw_section.get(key, {})
    if not isinstance(raw, dict):
        return {}

    out: dict[str, float] = {}
    for term, weight in raw.items():
        try:
            out[str(term).lower()] = float(weight)
        except (TypeError, ValueError):
            continue

    return out


def get_nested_terms(profile: dict[str, Any], section: str, key: str) -> list[str]:
    raw_section = profile.get(section, {})
    if not isinstance(raw_section, dict):
        return []

    raw = raw_section.get(key, [])
    if not isinstance(raw, list):
        return []

    return [str(x).lower() for x in raw]


def normalize_query(text: str, profile_name: str = "default") -> str:
    t = text.lower()
    profile = get_rerank_profile(profile_name)

    replacements = profile.get("query_normalizations", {})
    if not isinstance(replacements, dict):
        replacements = {}

    for src, dst in replacements.items():
        t = t.replace(str(src).lower(), str(dst).lower())

    return t


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def term_matches_query(term: str, q_norm: str, q_tokens: set[str]) -> bool:
    """
    Support both single-token terms and phrase terms such as 'how long'.
    """
    term = term.lower().strip()
    if not term:
        return False

    if " " in term:
        return term in q_norm

    return term in q_tokens


def term_matches_text(term: str, text_norm: str, t_tokens: set[str]) -> bool:
    """
    Support both single-token terms and phrase terms in retrieved text.
    """
    term = term.lower().strip()
    if not term:
        return False

    if " " in term:
        return term in text_norm

    return term in t_tokens


def query_has_any_term(terms: list[str], q_norm: str, q_tokens: set[str]) -> bool:
    return any(term_matches_query(term, q_norm, q_tokens) for term in terms)


def apply_weight_map_to_text(
    weights: dict[str, float],
    text_norm: str,
    t_tokens: set[str],
) -> float:
    bonus = 0.0

    for term, weight in weights.items():
        if term_matches_text(term, text_norm, t_tokens):
            bonus += weight

    return bonus


def numeric_boundary_bonus(query: str, text: str, profile_name: str = "default") -> float:
    profile = get_rerank_profile(profile_name)

    q_norm = normalize_query(query, profile_name=profile_name)
    text_norm = text.lower()

    q_tokens = set(tokenize(q_norm))
    t_tokens = set(tokenize(text_norm))

    duration_query_terms = get_nested_terms(
        profile, "numeric_boundary", "duration_query_terms"
    )
    duration_positive_terms = get_nested_weight_map(
        profile, "numeric_boundary", "duration_positive_terms"
    )
    duration_negative_terms = get_nested_weight_map(
        profile, "numeric_boundary", "duration_negative_terms"
    )

    value_query_terms = get_nested_terms(
        profile, "numeric_boundary", "value_query_terms"
    )
    value_positive_terms = get_nested_weight_map(
        profile, "numeric_boundary", "value_positive_terms"
    )

    bonus = 0.0

    asks_duration = query_has_any_term(duration_query_terms, q_norm, q_tokens)
    if asks_duration:
        bonus += apply_weight_map_to_text(duration_positive_terms, text_norm, t_tokens)
        bonus += apply_weight_map_to_text(duration_negative_terms, text_norm, t_tokens)

    asks_numeric_value = query_has_any_term(value_query_terms, q_norm, q_tokens)
    if asks_numeric_value:
        bonus += apply_weight_map_to_text(value_positive_terms, text_norm, t_tokens)

    return bonus


def lexical_bonus(query: str, text: str, profile_name: str = "default") -> float:
    profile = get_rerank_profile(profile_name)

    q_norm = normalize_query(query, profile_name=profile_name)
    text_norm = text.lower()

    q_tokens = set(tokenize(q_norm))
    t_tokens = set(tokenize(text_norm))

    bonus = 0.0

    # overlap bonus
    overlap = q_tokens & t_tokens
    bonus += 0.03 * len(overlap)

    # configurable term-weight bonus
    boost_terms = get_weight_map(profile, "boost_terms")
    for tok, weight in boost_terms.items():
        if tok in q_tokens and tok in t_tokens:
            bonus += weight

    # configurable numeric-boundary bonus
    bonus += numeric_boundary_bonus(query, text, profile_name=profile_name)

    return bonus


def rerank_score(query: str, result: dict[str, Any], profile_name: str = "default") -> float:
    """
    Higher is better.

    score = -cosine_distance + lexical_bonus(query, snippet)
    """
    dist = float(result.get("cosine_distance", 999.0))
    snippet = str(result.get("snippet", ""))
    return (-dist) + lexical_bonus(query, snippet, profile_name=profile_name)


def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    profile_name: str = "default",
) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []

    for r in results:
        x = dict(r)
        x["rerank_score"] = rerank_score(query, r, profile_name=profile_name)
        reranked.append(x)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


def rerank_rows(
    query: str,
    rows: list[tuple],
    profile_name: str = "default",
) -> list[tuple]:
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
    reranked = rerank_results(query, retrieved, profile_name=profile_name)

    row_by_chunk_id = {row[0]: row for row in rows}  # row[0] = chunk_id
    out: list[tuple] = []

    for r in reranked:
        cid = r.get("chunk_id")
        if cid in row_by_chunk_id:
            out.append(row_by_chunk_id[cid])

    return out
