import os
import time
import threading
import hashlib
from collections import OrderedDict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMS = int(os.getenv("OPENAI_EMBED_DIMS", "768"))

CACHE_MAX = int(os.getenv("EMBED_CACHE_MAX", "2000"))
CACHE_TTL = int(os.getenv("EMBED_CACHE_TTL_SECONDS", "3600"))

# key -> (ts, np.ndarray)
_cache: "OrderedDict[str, tuple[float, np.ndarray]]" = OrderedDict()
_lock = threading.Lock()


def _key_for_text(text: str) -> str:
    # normalize a little to reduce duplicates
    t = " ".join((text or "").strip().split())
    h = hashlib.sha256(t.encode("utf-8")).hexdigest()
    return f"{EMBED_MODEL}|{EMBED_DIMS}|{h}"


def _cache_get(key: str) -> np.ndarray | None:
    now = time.time()
    with _lock:
        item = _cache.get(key)
        if item is None:
            return None
        ts, vec = item
        if now - ts > CACHE_TTL:
            # expired
            _cache.pop(key, None)
            return None
        # LRU refresh
        _cache.move_to_end(key)
        return vec


def _cache_put(key: str, vec: np.ndarray) -> None:
    now = time.time()
    with _lock:
        _cache[key] = (now, vec)
        _cache.move_to_end(key)
        # evict LRU
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []

    # 1) cache lookup
    keys = [_key_for_text(t) for t in texts]
    out: list[np.ndarray | None] = [None] * len(texts)

    missing_texts: list[str] = []
    missing_pos: list[int] = []

    for i, k in enumerate(keys):
        v = _cache_get(k)
        if v is not None:
            out[i] = v
        else:
            missing_texts.append(texts[i])
            missing_pos.append(i)

    # 2) call OpenAI only for misses
    if missing_texts:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=missing_texts,
            dimensions=EMBED_DIMS,
            encoding_format="float",
        )
        new_vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]

        for pos, vec, text in zip(missing_pos, new_vecs, missing_texts):
            out[pos] = vec
            _cache_put(_key_for_text(text), vec)

    # 3) type assert
    return [v for v in out if v is not None]


def embed_queries(texts: list[str]) -> list[np.ndarray]:
    return embed_texts(texts)


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    return embed_texts(texts)
