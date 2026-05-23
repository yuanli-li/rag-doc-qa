from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class AskCompareCase:
    qid: str
    question: str
    document_id: int
    gold_chunk_id: int
    doc_name: str
    gold_page_or_section: str
    gold_chunk_hint: str


DEFAULT_CASES: list[AskCompareCase] = [
    AskCompareCase(
        qid="Q04",
        question="What is the purpose of the project?",
        document_id=4,
        gold_chunk_id=19,
        doc_name="test_eval_notes.md",
        gold_page_or_section="Project Overview",
        gold_chunk_hint="purpose of the project",
    ),
    AskCompareCase(
        qid="Q05",
        question="What limitation or current constraint of the system is described in the markdown?",
        document_id=4,
        gold_chunk_id=22,
        doc_name="test_eval_notes.md",
        gold_page_or_section="Current Limitations",
        gold_chunk_hint="current limitations include OCR / reranker / benchmark metrics absence",
    ),
    AskCompareCase(
        qid="Q01",
        question="What is the study objective?",
        document_id=5,
        gold_chunk_id=26,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="study objective compare Signal-K during stress and recovery",
    ),
    AskCompareCase(
        qid="H03",
        question="What measurement approach was chosen, and why was that choice helpful for evaluation?",
        document_id=5,
        gold_chunk_id=32,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="fluorescence assay and easy to retrieve",
    ),
    AskCompareCase(
        qid="H06",
        question="What was the baseline value of Signal-K before stress began?",
        document_id=5,
        gold_chunk_id=31,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="normalized to baseline before stress began; no numeric baseline value",
    ),
]


def get_loc_from_citation(c: dict[str, Any]) -> str:
    section = c.get("section")
    page = c.get("page")
    if section:
        return str(section)
    if page is not None:
        return f"page {page}"
    return "unknown"


def fetch_ask_result(base_url: str, endpoint: str, question: str, document_id: int, top_k: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    payload = {
        "query_text": question,
        "top_k": top_k,
        "document_id": document_id,
    }

    resp = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()

    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(f"{endpoint} returned non-ok response: {data}")

    return data


def gold_cited(data: dict[str, Any], gold_chunk_id: int) -> bool:
    used_ids = data.get("cited_chunk_ids", [])
    return gold_chunk_id in used_ids


def first_citation_loc(data: dict[str, Any]) -> str:
    citations = data.get("citations", [])
    if not citations:
        return "none"
    return get_loc_from_citation(citations[0])


def first_citation_chunk(data: dict[str, Any]) -> str:
    citations = data.get("citations", [])
    if not citations:
        return "none"
    return str(citations[0].get("chunk_id"))


def short_answer(text: str, limit: int = 160) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= limit:
        return t
    return t[:limit] + "..."


def summarize_comment(endpoint: str, data: dict[str, Any], gold_chunk_id: int) -> str:
    refused = data.get("refused", False)
    used_ids = data.get("cited_chunk_ids", [])
    if refused:
        return f"{endpoint} refused."
    if gold_chunk_id in used_ids:
        return f"{endpoint} cited the gold chunk."
    if used_ids:
        return f"{endpoint} cited non-gold chunk(s): {used_ids}."
    return f"{endpoint} returned no cited chunks."


def run_case(base_url: str, case: AskCompareCase, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for endpoint in ("ask", "ask_rerank"):
        data = fetch_ask_result(
            base_url=base_url,
            endpoint=endpoint,
            question=case.question,
            document_id=case.document_id,
            top_k=top_k,
        )

        rows.append(
            {
                "qid": case.qid,
                "endpoint": endpoint,
                "doc_name": case.doc_name,
                "question": case.question,
                "document_id": case.document_id,
                "gold_chunk_id": case.gold_chunk_id,
                "gold_page_or_section": case.gold_page_or_section,
                "gold_chunk_hint": case.gold_chunk_hint,
                "refused": data.get("refused", False),
                "first_citation_chunk_id": first_citation_chunk(data),
                "first_citation_loc": first_citation_loc(data),
                "cited_chunk_ids": ",".join(str(x) for x in data.get("cited_chunk_ids", [])),
                "gold_cited": "yes" if gold_cited(data, case.gold_chunk_id) else "no",
                "answer_preview": short_answer(data.get("answer", "")),
                "comment": summarize_comment(endpoint, data, case.gold_chunk_id),
            }
        )

    return rows


def write_tsv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "qid",
        "endpoint",
        "doc_name",
        "question",
        "document_id",
        "gold_chunk_id",
        "gold_page_or_section",
        "gold_chunk_hint",
        "refused",
        "first_citation_chunk_id",
        "first_citation_loc",
        "cited_chunk_ids",
        "gold_cited",
        "answer_preview",
        "comment",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    by_endpoint: dict[str, list[dict[str, Any]]] = {
        "ask": [], "ask_rerank": []}
    for r in rows:
        by_endpoint[r["endpoint"]].append(r)

    def gold_hits(endpoint: str) -> int:
        return sum(1 for r in by_endpoint[endpoint] if r["gold_cited"] == "yes")

    n_cases = len(by_endpoint["ask"])

    ask_hits = gold_hits("ask")
    ask_rerank_hits = gold_hits("ask_rerank")

    ask_map = {r["qid"]: r for r in by_endpoint["ask"]}
    rerank_map = {r["qid"]: r for r in by_endpoint["ask_rerank"]}

    improved = 0
    worsened = 0
    unchanged_good = 0
    unchanged_bad = 0

    for qid, r0 in ask_map.items():
        r1 = rerank_map[qid]
        before = r0["gold_cited"] == "yes"
        after = r1["gold_cited"] == "yes"

        if (not before) and after:
            improved += 1
        elif before and (not after):
            worsened += 1
        elif before and after:
            unchanged_good += 1
        else:
            unchanged_bad += 1

    print("\n=== Ask A/B Summary ===")
    print(f"Cases evaluated:            {n_cases}")
    print(f"ask gold-citation hits:     {ask_hits}/{n_cases}")
    print(f"ask_rerank gold-citation hits: {ask_rerank_hits}/{n_cases}")
    print(f"improved cases:             {improved}")
    print(f"worsened cases:             {worsened}")
    print(f"unchanged good cases:       {unchanged_good}")
    print(f"unchanged bad cases:        {unchanged_bad}")

    print("\n=== Per-case delta ===")
    for qid in sorted(ask_map.keys()):
        r0 = ask_map[qid]
        r1 = rerank_map[qid]
        print(
            f"{qid}: "
            f"ask(gold_cited={r0['gold_cited']}, first_citation={r0['first_citation_chunk_id']}, refused={r0['refused']}) -> "
            f"ask_rerank(gold_cited={r1['gold_cited']}, first_citation={r1['first_citation_chunk_id']}, refused={r1['refused']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run /ask vs /ask_rerank A/B comparison over a fixed question set.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="Base URL for the FastAPI app")
    parser.add_argument("--top-k", type=int, default=4,
                        help="top_k to request from the API")
    parser.add_argument(
        "--out",
        default="eval/data/results/ask_ab_compare.tsv",
        help="Output TSV path",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []

    print("Running ask A/B comparison...")
    for case in DEFAULT_CASES:
        print(f"  - {case.qid}: {case.question}")
        try:
            case_rows = run_case(args.base_url, case, args.top_k)
            all_rows.extend(case_rows)
        except requests.HTTPError as e:
            print(f"[ERROR] HTTP error for {case.qid}: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"[ERROR] Failed on {case.qid}: {e}", file=sys.stderr)
            return 1

    out_path = Path(args.out)
    write_tsv(all_rows, out_path)
    print_summary(all_rows)
    print(f"\nSaved TSV to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
