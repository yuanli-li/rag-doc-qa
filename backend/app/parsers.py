import io
from typing import List, Dict, Any

# 尝试导入更强大的 PyMuPDF (fitz)
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# 保留 pypdf 作为备选
from pypdf import PdfReader


def parse_pdf(data: bytes) -> List[Dict[str, Any]]:
    """
    解析 PDF：优先使用 PyMuPDF 处理学术论文的双栏布局，
    如果未安装则降级使用 pypdf。
    """
    units = []

    if PYMUPDF_AVAILABLE:
        # --- 方案 A: 使用 PyMuPDF (更适合双栏论文) ---
        with fitz.open(stream=data, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                # "text" 模式能够根据阅读流智能识别分栏顺序
                text = page.get_text("text").strip()
                if text:
                    # 将多个空格和换行符统一成一个空格，清理碎片化文本
                    clean_text = " ".join(text.split())
                    units.append({
                        "text": clean_text,
                        "meta": {"page": i + 1}
                    })
    else:
        # --- 方案 B: 使用 pypdf (基础 fallback) ---
        reader = PdfReader(io.BytesIO(data))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                clean_text = " ".join(text.split())
                units.append({
                    "text": clean_text,
                    "meta": {"page": i + 1}
                })

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
