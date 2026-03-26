import hashlib
from typing import List, Dict, Any, Tuple
from psycopg.types.json import Jsonb

from .db import get_conn


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def upsert_document(
    title: str,
    source: str,
    filename: str | None,
    mime_type: str | None,
    sha256: str,
    replace: bool,
) -> int:
    """
    If a doc with same sha256 exists:
      - replace=True: delete its chunks and reuse doc row
      - replace=False: return existing doc id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE sha256 = %s LIMIT 1;", (sha256,))
            row = cur.fetchone()
            if row:
                doc_id = row[0]
                if replace:
                    cur.execute(
                        "DELETE FROM chunks WHERE document_id = %s;", (doc_id,))
                    cur.execute(
                        """
                        UPDATE documents
                        SET title=%s, source=%s, filename=%s, mime_type=%s
                        WHERE id=%s;
                        """,
                        (title, source, filename, mime_type, doc_id),
                    )
                return doc_id

            cur.execute(
                """
                INSERT INTO documents (title, source, filename, mime_type, sha256)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (title, source, filename, mime_type, sha256),
            )
            return cur.fetchone()[0]


def insert_chunks(
    document_id: int,
    chunks: List[Dict[str, Any]],
    vectors,  # list[np.ndarray]
) -> int:
    """
    chunks: [{"text":..., "meta":...}, ...]
    vectors: same length list of np arrays (768-d)
    """
    assert len(chunks) == len(vectors)

    with get_conn() as conn:
        with conn.cursor() as cur:
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (document_id, idx, chunk["text"],
                     Jsonb(chunk.get("meta", {})), vec),
                )
    return len(chunks)
