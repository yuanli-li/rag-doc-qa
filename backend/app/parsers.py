# backend/app/parsers.py
from __future__ import annotations
from typing import Any, Dict, List

import re


def _clean_text(t: str) -> str:
    if not t:
        return ""
    # 常见断词：hyphen + 换行
    t = re.sub(r"-\s*\n\s*", "", t)
    # 把单换行压成空格（保留段落空行）
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # 多空白压缩
    t = re.sub(r"[ \t]+", " ", t)
    # 多空行压缩到最多两个
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def parse_pdf(data: bytes) -> List[Dict[str, Any]]:
    """
    Return units: [{ "text": str, "meta": {"page": int} }, ...]
    Page is 1-indexed.
    """
    units: List[Dict[str, Any]] = []

    # 1) Prefer PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")

        for i in range(len(doc)):
            page = doc[i]
            # (x0,y0,x1,y1,text,block_no,block_type)
            blocks = page.get_text("blocks")

            # keep non-empty text blocks
            text_blocks = []
            for b in blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                if not text or not text.strip():
                    continue
                text_blocks.append((x0, y0, x1, y1, text))

            if not text_blocks:
                continue

            # 2) simple 2-column aware ordering:
            #    left column top->down, then right column top->down
            w = page.rect.width
            mid = w * 0.5

            left = [b for b in text_blocks if b[0] <= mid]
            right = [b for b in text_blocks if b[0] > mid]

            left.sort(key=lambda t: (t[1], t[0]))   # y0 then x0
            right.sort(key=lambda t: (t[1], t[0]))

            ordered = left + right if right else left

            page_text = "\n".join([b[4] for b in ordered])
            page_text = _clean_text(page_text)
            if not page_text:
                continue

            # 3) split into paragraph-like units (better for chunking)
            paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            for p in paras:
                units.append({"text": p, "meta": {"page": i + 1}})

        return units

    except Exception:
        # 4) fallback to pypdf (old behavior)
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(data))
            for i, p in enumerate(reader.pages):
                t = p.extract_text() or ""
                t = _clean_text(t)
                if not t:
                    continue
                paras = [x.strip() for x in t.split("\n\n") if x.strip()]
                for para in paras:
                    units.append({"text": para, "meta": {"page": i + 1}})
            return units
        except Exception:
            return []


def parse_md(data: bytes | str) -> list[dict]:
    """
    Return units: [{ "text": str, "meta": {"section": str, "level": int} }, ...]
    """
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="ignore")
    else:
        text = data

    lines = text.splitlines()

    units = []
    cur_section = "Document"
    cur_level = 0
    buf = []

    def flush():
        nonlocal buf
        content = "\n".join(buf).strip()
        if content:
            units.append({
                "text": content,
                "meta": {"section": cur_section, "level": cur_level}
            })
        buf = []

    for line in lines:
        # markdown heading
        if line.startswith("#"):
            m = re.match(r"^(#+)\s+(.*)$", line.strip())
            if m:
                flush()
                hashes, title = m.group(1), m.group(2).strip()
                cur_section = title if title else "Untitled"
                cur_level = len(hashes)
                continue

        buf.append(line)

    flush()
    # If no headings, put everything as one section
    if not units and text.strip():
        units.append({"text": text.strip(), "meta": {
                     "section": "Document", "level": 0}})
    return units
