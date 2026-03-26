import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-embedding-001"  # stable text embedding model
OUTPUT_DIM = 768               # recommended size for storage/perf tradeoff
EXPECTED_DIM = OUTPUT_DIM

_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Put it in backend/.env")

_client = genai.Client(api_key=_API_KEY)


def embed_texts(texts: list[str], task_type: str) -> list[np.ndarray]:
    """
    task_type:
      - "RETRIEVAL_DOCUMENT" for chunks
      - "RETRIEVAL_QUERY" for query
    """
    clean = [t.strip() for t in texts]
    if any(len(t) == 0 for t in clean):
        raise ValueError("Empty string is not allowed for embeddings input.")

    resp = _client.models.embed_content(
        model=MODEL,
        contents=clean,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=OUTPUT_DIM,
        ),
    )

    vectors: list[np.ndarray] = []
    for emb in resp.embeddings:
        v = np.array(emb.values, dtype=np.float32)
        if v.shape[0] != EXPECTED_DIM:
            raise RuntimeError(
                f"Embedding dim mismatch: got {v.shape[0]}, expected {EXPECTED_DIM}")
        vectors.append(v)

    return vectors


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_queries(texts: list[str]) -> list[np.ndarray]:
    return embed_texts(texts, task_type="RETRIEVAL_QUERY")
