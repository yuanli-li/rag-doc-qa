import io
from typing import List, Dict, Any
from pypdf import PdfReader


def parse_pdf(data: bytes) -> List[Dict[str, Any]]:
    """
    Returns units per page:
      [{"text": "...", "meta": {"page": 1}}, ...]
    Notes:
    - If PDF is scanned/image-only, extract_text may return None/empty.
    """
    reader = PdfReader(io.BytesIO(data))
    units: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            units.append({"text": text, "meta": {"page": i + 1}})
    return units


def parse_md(data: bytes) -> List[Dict[str, Any]]:
    """
    Split Markdown by headings (#, ##, ### ...).
    Returns units:
      [{"text": "...", "meta": {"section": "Intro", "level": 1}}, ...]
    """
    text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    units: List[Dict[str, Any]] = []
    cur_title = "root"
    cur_level = 0
    buf: List[str] = []

    def flush():
        nonlocal buf, cur_title, cur_level
        content = "\n".join(buf).strip()
        if content:
            units.append({"text": content, "meta": {
                         "section": cur_title, "level": cur_level}})
        buf = []

    for line in lines:
        if line.startswith("#"):
            # new section
            flush()
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip() or "untitled"
            cur_title, cur_level = title, level
        else:
            buf.append(line)

    flush()
    return units
