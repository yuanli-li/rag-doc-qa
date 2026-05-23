from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class CompareCase:
    qid: str
    question: str
    document_id: int
    gold_chunk_id: int
    doc_name: str
    gold_page_or_section: str
    gold_chunk_hint: str


# 你当前已经跑过、并且有 gold chunk id 的题
DEFAULT_CASES: list[CompareCase] = [
    CompareCase(
        qid="Q04",
        question="What is the purpose of the project?",
        document_id=4,
        gold_chunk_id=19,
        doc_name="test_eval_notes.md",
        gold_page_or_section="Project Overview",
        gold_chunk_hint="purpose of the project",
    ),
    CompareCase(
        qid="Q05",
        question="What limitation or current constraint of the system is described in the markdown?",
        document_id=4,
        gold_chunk_id=22,
        doc_name="test_eval_notes.md",
        gold_page_or_section="Current Limitations",
        gold_chunk_hint="current limitations include OCR / reranker / benchmark metrics absence",
    ),
    CompareCase(
        qid="Q01",
        question="What is the study objective?",
        document_id=5,
        gold_chunk_id=26,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="study objective compare Signal-K during stress and recovery",
    ),
    CompareCase(
        qid="H03",
        question="What measurement approach was chosen, and why was that choice helpful for evaluation?",
        document_id=5,
        gold_chunk_id=32,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="fluorescence assay and easy to retrieve",
    ),
    CompareCase(
        qid="H06",
        question="What was the baseline value of Signal-K before stress began?",
        document_id=5,
        gold_chunk_id=31,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="normalized to baseline before stress began; no numeric baseline value",
    ),
    CompareCase(
        qid="Q02",
        question="What assay was used to quantify Signal-K?",
        document_id=5,
        gold_chunk_id=32,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="fluorescence assay",
    ),
    CompareCase(
        qid="T08",
        question="Which passage is the strongest evidence that no benchmark score was reported?",
        document_id=4,
        gold_chunk_id=22,
        doc_name="test_eval_notes.md",
        gold_page_or_section="Current Limitations",
        gold_chunk_hint="does not report benchmark metrics such as F1, MRR, or accuracy",
    ),
]


def get_loc(result: dict[str, Any]) -> str:
    section = result.get("section")
    page = result.get("page")
    if section:
        return str(section)
    if page is not None:
        return f"page {page}"
    return "unknown"


def fetch_results(base_url: str, endpoint: str, question: str, document_id: int, top_k: int) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    payload = {
        "query_text": question,
        "top_k": top_k,
        "document_id": document_id,
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(
            f"{endpoint} returned non-ok response: {json.dumps(data, ensure_ascii=False)}")

    results = data.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError(
            f"{endpoint} results is not a list: {json.dumps(data, ensure_ascii=False)}")

    return results


def get_gold_rank(results: list[dict[str, Any]], gold_chunk_id: int) -> int | None:
    for idx, r in enumerate(results, start=1):
        if r.get("chunk_id") == gold_chunk_id:
            return idx
    return None


def summarize_comment(endpoint: str, top1_chunk_id: int | None, gold_rank: int | None, top1_is_gold: str) -> str:
    if top1_is_gold == "yes":
        return f"{endpoint} top1 matched gold."
    if gold_rank is None:
        return f"{endpoint} did not retrieve gold in top_k."
    return f"{endpoint} gold was rank {gold_rank}, not top1."


def run_case(base_url: str, case: CompareCase, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for endpoint in ("retrieve", "retrieve_rerank"):
        results = fetch_results(
            base_url=base_url,
            endpoint=endpoint,
            question=case.question,
            document_id=case.document_id,
            top_k=top_k,
        )

        top1 = results[0] if results else {}
        top1_chunk_id = top1.get("chunk_id")
        top1_loc = get_loc(top1) if top1 else "none"

        gold_rank = get_gold_rank(results, case.gold_chunk_id)
        top1_is_gold = "yes" if top1_chunk_id == case.gold_chunk_id else "no"

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
                "top1_chunk_id": top1_chunk_id,
                "top1_section_or_page": top1_loc,
                "gold_rank": gold_rank if gold_rank is not None else "NA",
                "top1_is_gold": top1_is_gold,
                "comment": summarize_comment(endpoint, top1_chunk_id, gold_rank, top1_is_gold),
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
        "top1_chunk_id",
        "top1_section_or_page",
        "gold_rank",
        "top1_is_gold",
        "comment",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    by_endpoint: dict[str, list[dict[str, Any]]] = {
        "retrieve": [], "retrieve_rerank": []}
    for r in rows:
        by_endpoint[r["endpoint"]].append(r)

    def top1_hits(endpoint: str) -> int:
        return sum(1 for r in by_endpoint[endpoint] if r["top1_is_gold"] == "yes")

    n_cases = len(by_endpoint["retrieve"])

    retrieve_hits = top1_hits("retrieve")
    rerank_hits = top1_hits("retrieve_rerank")

    # pair rows by qid
    retrieve_map = {r["qid"]: r for r in by_endpoint["retrieve"]}
    rerank_map = {r["qid"]: r for r in by_endpoint["retrieve_rerank"]}

    repaired = 0
    worsened = 0
    unchanged_good = 0
    unchanged_bad = 0

    for qid, r0 in retrieve_map.items():
        r1 = rerank_map[qid]
        before = r0["top1_is_gold"] == "yes"
        after = r1["top1_is_gold"] == "yes"

        if (not before) and after:
            repaired += 1
        elif before and (not after):
            worsened += 1
        elif before and after:
            unchanged_good += 1
        else:
            unchanged_bad += 1

    print("\n=== Retrieve A/B Summary ===")
    print(f"Cases evaluated:           {n_cases}")
    print(f"retrieve top1 hits:        {retrieve_hits}/{n_cases}")
    print(f"retrieve_rerank top1 hits: {rerank_hits}/{n_cases}")
    print(f"repaired cases:            {repaired}")
    print(f"worsened cases:            {worsened}")
    print(f"unchanged good cases:      {unchanged_good}")
    print(f"unchanged bad cases:       {unchanged_bad}")

    print("\n=== Per-case delta ===")
    for qid in sorted(retrieve_map.keys()):
        r0 = retrieve_map[qid]
        r1 = rerank_map[qid]
        print(
            f"{qid}: "
            f"retrieve(top1={r0['top1_chunk_id']}, gold_rank={r0['gold_rank']}, top1_is_gold={r0['top1_is_gold']}) -> "
            f"retrieve_rerank(top1={r1['top1_chunk_id']}, gold_rank={r1['gold_rank']}, top1_is_gold={r1['top1_is_gold']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run retrieve vs retrieve_rerank A/B comparison over a fixed question set.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="Base URL for the FastAPI app")
    parser.add_argument("--top-k", type=int, default=4,
                        help="top_k to request from the API")
    parser.add_argument(
        "--out",
        default="eval/data/results/retrieve_ab_compare.tsv",
        help="Output TSV path",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []

    print("Running retrieve A/B comparison...")
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
