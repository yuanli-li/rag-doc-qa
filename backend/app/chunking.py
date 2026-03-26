from typing import List, Dict, Any, Tuple


def chunk_units(
    units: List[Dict[str, Any]],
    max_words: int = 250,
    overlap_words: int = 50,
) -> List[Dict[str, Any]]:
    """
    Convert units (text + meta) into chunk list:
      [{"text": "...", "meta": {...}}, ...]
    Strategy:
      - word-based sliding window per unit
      - keeps unit meta (page/section) on each chunk
    """
    chunks: List[Dict[str, Any]] = []

    for unit in units:
        text = unit["text"]
        meta = unit.get("meta", {})
        words = text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "meta": dict(meta)})
            if end == len(words):
                break
            start = max(0, end - overlap_words)

    return chunks
