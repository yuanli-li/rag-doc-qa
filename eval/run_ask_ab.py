from __future__ import annotations

import argparse
import csv
import re
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
    question_type: str = ""
    subset: str = ""
    acceptable_chunk_ids: tuple[int, ...] = ()


DEFAULT_CASES: list[AskCompareCase] = [
    AskCompareCase(
        qid="Q04",
        question="What is the purpose of the project?",
        document_id=4,
        gold_chunk_id=19,
        doc_name="test_eval_notes.md",
        gold_page_or_section="Project Overview",
        gold_chunk_hint="purpose of the project",
        question_type="purpose_objective",
        subset="default",
        acceptable_chunk_ids=(19,),
    ),
    AskCompareCase(
        qid="Q05",
        question="What limitation or current constraint of the system is described in the markdown?",
        document_id=4,
        gold_chunk_id=22,
        doc_name="test_eval_notes.md",
        gold_page_or_section="Current Limitations",
        gold_chunk_hint="current limitations include OCR / reranker / benchmark metrics absence",
        question_type="negative_evidence",
        subset="default",
        acceptable_chunk_ids=(22,),
    ),
    AskCompareCase(
        qid="Q01",
        question="What is the study objective?",
        document_id=5,
        gold_chunk_id=26,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="study objective compare Signal-K during stress and recovery",
        question_type="purpose_objective",
        subset="default",
        acceptable_chunk_ids=(26, 28),
    ),
    AskCompareCase(
        qid="H03",
        question="What measurement approach was chosen, and why was that choice helpful for evaluation?",
        document_id=5,
        gold_chunk_id=32,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="fluorescence assay and easy to retrieve",
        question_type="method_explanation",
        subset="default",
        acceptable_chunk_ids=(32,),
    ),
    AskCompareCase(
        qid="H06",
        question="What was the baseline value of Signal-K before stress began?",
        document_id=5,
        gold_chunk_id=31,
        doc_name="test_dual_column_eval.pdf",
        gold_page_or_section="page 1",
        gold_chunk_hint="normalized to baseline before stress began; no numeric baseline value",
        question_type="numeric_boundary",
        subset="default",
        acceptable_chunk_ids=(31,),
    ),
]


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Strip whitespace from DictReader keys and string values.

    This makes the script robust to TSV headers like:
      ' document_id'
    instead of:
      'document_id'
    """
    return {
        (k.strip() if k is not None else k): (v.strip() if isinstance(v, str) else v)
        for k, v in row.items()
    }


def parse_acceptable_chunk_ids(clean: dict[str, Any]) -> tuple[int, ...]:
    """
    Parse optional acceptable_chunk_ids column.

    If the column is absent or empty, fall back to gold_chunk_id.
    Example:
      gold_chunk_id=26, acceptable_chunk_ids=26,28
    means either chunk 26 or 28 is acceptable at answer/citation level.
    """
    raw = clean.get("acceptable_chunk_ids", "")

    if isinstance(raw, str) and raw.strip():
        ids: list[int] = []
        for x in raw.split(","):
            x = x.strip()
            if x:
                ids.append(int(x))
        return tuple(ids)

    return (int(clean["gold_chunk_id"]),)


def read_questions(path: Path) -> list[AskCompareCase]:
    cases: list[AskCompareCase] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        required = {
            "qid",
            "question",
            "document_id",
            "doc_name",
            "gold_chunk_id",
            "gold_page_or_section",
            "gold_chunk_hint",
            "question_type",
            "subset",
        }

        if reader.fieldnames is None:
            raise ValueError(f"No header found in questions TSV: {path}")

        normalized_fieldnames = {x.strip() for x in reader.fieldnames}
        missing = required - normalized_fieldnames
        if missing:
            raise ValueError(
                f"Questions TSV is missing required column(s): {sorted(missing)}\n"
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            clean = clean_row(row)

            # Skip fully blank lines defensively.
            if not clean.get("qid") and not clean.get("question"):
                continue

            cases.append(
                AskCompareCase(
                    qid=clean["qid"],
                    question=clean["question"],
                    document_id=int(clean["document_id"]),
                    gold_chunk_id=int(clean["gold_chunk_id"]),
                    doc_name=clean["doc_name"],
                    gold_page_or_section=clean["gold_page_or_section"],
                    gold_chunk_hint=clean["gold_chunk_hint"],
                    question_type=clean.get("question_type", ""),
                    subset=clean.get("subset", ""),
                    acceptable_chunk_ids=parse_acceptable_chunk_ids(clean),
                )
            )

    return cases


def filter_cases(
    cases: list[AskCompareCase],
    *,
    subset: str | None = None,
    qid: str | None = None,
    question_type: str | None = None,
    doc: str | None = None,
    contains: str | None = None,
) -> list[AskCompareCase]:
    out: list[AskCompareCase] = []

    for case in cases:
        if subset is not None and case.subset != subset:
            continue

        if qid is not None and case.qid != qid:
            continue

        if question_type is not None and case.question_type != question_type:
            continue

        if doc is not None and doc not in case.doc_name:
            continue

        if contains is not None and contains.lower() not in case.question.lower():
            continue

        out.append(case)

    return out


def safe_label(text: str) -> str:
    """
    Convert labels such as 'phase1_hard' or 'numeric/boundary'
    into safe filename fragments.
    """
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "filtered"


def build_default_out_path(args: argparse.Namespace) -> Path:
    """
    If user did not explicitly pass --out, create an informative output path.
    """
    default_out = "eval/data/results/ask_ab_compare.tsv"

    if args.out != default_out:
        return Path(args.out)

    parts = ["ask_ab_compare"]

    if args.subset:
        parts.append(f"subset_{safe_label(args.subset)}")
    if args.qid:
        parts.append(f"qid_{safe_label(args.qid)}")
    if args.question_type:
        parts.append(f"qtype_{safe_label(args.question_type)}")
    if args.doc:
        parts.append(f"doc_{safe_label(args.doc)}")
    if args.contains:
        parts.append(f"contains_{safe_label(args.contains)}")

    if len(parts) == 1:
        return Path(default_out)

    return Path("eval/data/results") / ("_".join(parts) + ".tsv")


def get_loc_from_citation(c: dict[str, Any]) -> str:
    section = c.get("section")
    page = c.get("page")
    if section:
        return str(section)
    if page is not None:
        return f"page {page}"
    return "unknown"


def fetch_ask_result(
    base_url: str,
    endpoint: str,
    question: str,
    document_id: int,
    top_k: int,
) -> dict[str, Any]:
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


def get_acceptable_chunk_ids(case: AskCompareCase) -> tuple[int, ...]:
    if case.acceptable_chunk_ids:
        return case.acceptable_chunk_ids
    return (case.gold_chunk_id,)


def gold_cited(data: dict[str, Any], acceptable_chunk_ids: tuple[int, ...]) -> bool:
    used_ids = set(data.get("cited_chunk_ids", []))
    acceptable = set(acceptable_chunk_ids)
    return bool(used_ids & acceptable)


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


def summarize_comment(
    endpoint: str,
    data: dict[str, Any],
    acceptable_chunk_ids: tuple[int, ...],
) -> str:
    refused = data.get("refused", False)
    used_ids = set(data.get("cited_chunk_ids", []))
    acceptable = set(acceptable_chunk_ids)

    if refused:
        return f"{endpoint} refused."

    if used_ids & acceptable:
        matched = sorted(used_ids & acceptable)
        return f"{endpoint} cited acceptable chunk(s): {matched}."

    if used_ids:
        return f"{endpoint} cited non-acceptable chunk(s): {sorted(used_ids)}."

    return f"{endpoint} returned no cited chunks."


def run_case(base_url: str, case: AskCompareCase, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    acceptable_ids = get_acceptable_chunk_ids(case)

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
                "acceptable_chunk_ids": ",".join(str(x) for x in acceptable_ids),
                "gold_page_or_section": case.gold_page_or_section,
                "gold_chunk_hint": case.gold_chunk_hint,
                "question_type": case.question_type,
                "subset": case.subset,
                "refused": data.get("refused", False),
                "first_citation_chunk_id": first_citation_chunk(data),
                "first_citation_loc": first_citation_loc(data),
                "cited_chunk_ids": ",".join(str(x) for x in data.get("cited_chunk_ids", [])),
                "gold_cited": "yes" if gold_cited(data, acceptable_ids) else "no",
                "answer_preview": short_answer(data.get("answer", "")),
                "comment": summarize_comment(endpoint, data, acceptable_ids),
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
        "acceptable_chunk_ids",
        "gold_page_or_section",
        "gold_chunk_hint",
        "question_type",
        "subset",
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
        "ask": [],
        "ask_rerank": [],
    }

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
    print(f"Cases evaluated:               {n_cases}")
    print(f"ask gold-citation hits:        {ask_hits}/{n_cases}")
    print(f"ask_rerank gold-citation hits: {ask_rerank_hits}/{n_cases}")
    print(f"improved cases:                {improved}")
    print(f"worsened cases:                {worsened}")
    print(f"unchanged good cases:          {unchanged_good}")
    print(f"unchanged bad cases:           {unchanged_bad}")

    qtypes = sorted({r.get("question_type", "") for r in rows if r.get("question_type", "")})
    if qtypes:
        print("\n=== Ask A/B breakdown by question_type ===")
        for qt in qtypes:
            ask_qt = [r for r in by_endpoint["ask"] if r.get("question_type") == qt]
            rerank_qt = [r for r in by_endpoint["ask_rerank"] if r.get("question_type") == qt]

            if not ask_qt:
                continue

            ask_qt_hits = sum(1 for r in ask_qt if r["gold_cited"] == "yes")
            rerank_qt_hits = sum(1 for r in rerank_qt if r["gold_cited"] == "yes")

            print(
                f"{qt}: "
                f"ask={ask_qt_hits}/{len(ask_qt)} -> "
                f"ask_rerank={rerank_qt_hits}/{len(rerank_qt)}"
            )

    print("\n=== Per-case delta ===")
    for qid in sorted(ask_map.keys()):
        r0 = ask_map[qid]
        r1 = rerank_map[qid]
        print(
            f"{qid}: "
            f"ask(gold_cited={r0['gold_cited']}, "
            f"first_citation={r0['first_citation_chunk_id']}, "
            f"cited={r0['cited_chunk_ids']}, "
            f"acceptable={r0['acceptable_chunk_ids']}, "
            f"refused={r0['refused']}) -> "
            f"ask_rerank(gold_cited={r1['gold_cited']}, "
            f"first_citation={r1['first_citation_chunk_id']}, "
            f"cited={r1['cited_chunk_ids']}, "
            f"acceptable={r1['acceptable_chunk_ids']}, "
            f"refused={r1['refused']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run /ask vs /ask_rerank A/B comparison over eval questions."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the FastAPI app",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="top_k to request from the API",
    )
    parser.add_argument(
        "--out",
        default="eval/data/results/ask_ab_compare.tsv",
        help="Output TSV path",
    )
    parser.add_argument(
        "--questions",
        default="eval/data/retrieval_eval_questions.tsv",
        help="Path to retrieval eval questions TSV",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Only run cases from this subset",
    )
    parser.add_argument(
        "--qid",
        default=None,
        help="Only run one qid",
    )
    parser.add_argument(
        "--question-type",
        default=None,
        help="Only run one question_type",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help="Only run cases whose doc_name contains this string",
    )
    parser.add_argument(
        "--contains",
        default=None,
        help="Only run cases whose question contains this string",
    )

    args = parser.parse_args()

    questions_path = Path(args.questions)

    if questions_path.exists():
        cases = read_questions(questions_path)
        cases = filter_cases(
            cases,
            subset=args.subset,
            qid=args.qid,
            question_type=args.question_type,
            doc=args.doc,
            contains=args.contains,
        )
    else:
        print(
            f"[WARN] Questions TSV not found at {questions_path}. Falling back to DEFAULT_CASES.",
            file=sys.stderr,
        )
        cases = DEFAULT_CASES
        cases = filter_cases(
            cases,
            subset=args.subset,
            qid=args.qid,
            question_type=args.question_type,
            doc=args.doc,
            contains=args.contains,
        )

    if not cases:
        print("[ERROR] No cases matched the requested filters.", file=sys.stderr)
        return 1

    all_rows: list[dict[str, Any]] = []

    print("Running ask A/B comparison...")
    for case in cases:
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

    out_path = build_default_out_path(args)
    write_tsv(all_rows, out_path)
    print_summary(all_rows)
    print(f"\nSaved TSV to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
