from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import requests


def read_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["document_id"] = int(row["document_id"])
            row["gold_chunk_id"] = int(row["gold_chunk_id"])

            # allow backward compatibility if these columns are missing
            if "question_type" not in row or row["question_type"] is None:
                row["question_type"] = ""
            if "subset" not in row or row["subset"] is None:
                row["subset"] = ""

            rows.append(row)
    return rows


def filter_questions(
    questions: list[dict[str, Any]],
    qid: str | None = None,
    doc: str | None = None,
    contains: str | None = None,
    question_type: str | None = None,
    subset: str | None = None,
) -> list[dict[str, Any]]:
    out = questions

    if qid:
        qid_norm = qid.strip().lower()
        out = [q for q in out if q["qid"].strip().lower() == qid_norm]

    if doc:
        doc_norm = doc.strip().lower()
        out = [q for q in out if q["doc_name"].strip().lower() == doc_norm]

    if contains:
        needle = contains.strip().lower()
        out = [q for q in out if needle in q["question"].lower()]

    if question_type:
        qt_norm = question_type.strip().lower()
        out = [q for q in out if q.get(
            "question_type", "").strip().lower() == qt_norm]

    if subset:
        subset_norm = subset.strip().lower()
        out = [q for q in out if q.get(
            "subset", "").strip().lower() == subset_norm]

    return out


def fetch_results(
    base_url: str,
    endpoint: str,
    question: str,
    document_id: int,
    top_k: int,
) -> list[dict[str, Any]]:
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
        raise RuntimeError(f"{endpoint} returned non-ok response: {data}")

    results = data.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError(f"{endpoint} results is not a list: {data}")

    return results


def get_gold_rank(results: list[dict[str, Any]], gold_chunk_id: int) -> int | None:
    for idx, r in enumerate(results, start=1):
        if r.get("chunk_id") == gold_chunk_id:
            return idx
    return None


def get_loc(result: dict[str, Any]) -> str:
    section = result.get("section")
    page = result.get("page")
    if section:
        return str(section)
    if page is not None:
        return f"page {page}"
    return "unknown"


def summarize_comment(top1_is_gold: str, gold_rank: int | None) -> str:
    if top1_is_gold == "yes":
        return "Top1 matched gold."
    if gold_rank is None:
        return "Gold not found in top_k."
    return f"Gold was rank {gold_rank}, not top1."


def evaluate_endpoint(
    questions: list[dict[str, Any]],
    base_url: str,
    endpoint: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []

    print(f"Running retrieval eval using endpoint: {endpoint}")
    for q in questions:
        print(f"  - {q['qid']}: {q['question']}")
        try:
            results = fetch_results(
                base_url=base_url,
                endpoint=endpoint,
                question=q["question"],
                document_id=q["document_id"],
                top_k=top_k,
            )
        except requests.HTTPError as e:
            print(f"[ERROR] HTTP error for {q['qid']}: {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"[ERROR] Failed on {q['qid']}: {e}", file=sys.stderr)
            raise

        top1 = results[0] if results else {}
        top1_chunk_id = top1.get("chunk_id", "none")
        top1_loc = get_loc(top1) if top1 else "none"

        gold_rank = get_gold_rank(results, q["gold_chunk_id"])
        top1_is_gold = "yes" if top1_chunk_id == q["gold_chunk_id"] else "no"

        rows_out.append(
            {
                "qid": q["qid"],
                "endpoint": endpoint,
                "doc_name": q["doc_name"],
                "question": q["question"],
                "document_id": q["document_id"],
                "gold_chunk_id": q["gold_chunk_id"],
                "gold_page_or_section": q["gold_page_or_section"],
                "gold_chunk_hint": q["gold_chunk_hint"],
                "question_type": q.get("question_type", ""),
                "subset": q.get("subset", ""),
                "top1_chunk_id": top1_chunk_id,
                "top1_section_or_page": top1_loc,
                "gold_rank": gold_rank if gold_rank is not None else "NA",
                "top1_is_gold": top1_is_gold,
                "comment": summarize_comment(top1_is_gold, gold_rank),
            }
        )

    return rows_out


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
        "question_type",
        "subset",
        "top1_chunk_id",
        "top1_section_or_page",
        "gold_rank",
        "top1_is_gold",
        "comment",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_single_summary(rows: list[dict[str, Any]], endpoint: str) -> None:
    n = len(rows)
    top1_hits = sum(1 for r in rows if r["top1_is_gold"] == "yes")
    gold_in_topk = sum(1 for r in rows if r["gold_rank"] != "NA")

    print("\n=== Retrieval Eval Summary ===")
    print(f"Endpoint:                 {endpoint}")
    print(f"Cases evaluated:          {n}")
    print(f"gold_in_top_k:            {gold_in_topk}/{n}")
    print(f"top1 hits:                {top1_hits}/{n}")

    # optional breakdown by question_type
    type_counts: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        qt = r.get("question_type", "") or "(untyped)"
        type_counts.setdefault(qt, []).append(r)

    if len(type_counts) > 1:
        print("\n=== Breakdown by question_type ===")
        for qt, group in sorted(type_counts.items()):
            hits = sum(1 for r in group if r["top1_is_gold"] == "yes")
            print(f"{qt}: {hits}/{len(group)}")

    print("\n=== Per-case results ===")
    for r in rows:
        print(
            f"{r['qid']}: "
            f"top1={r['top1_chunk_id']} | "
            f"gold_rank={r['gold_rank']} | "
            f"top1_is_gold={r['top1_is_gold']}"
        )


def print_compare_summary(rows: list[dict[str, Any]]) -> None:
    retrieve_rows = [r for r in rows if r["endpoint"] == "retrieve"]
    rerank_rows = [r for r in rows if r["endpoint"] == "retrieve_rerank"]

    retrieve_map = {r["qid"]: r for r in retrieve_rows}
    rerank_map = {r["qid"]: r for r in rerank_rows}

    n = len(retrieve_rows)
    retrieve_hits = sum(1 for r in retrieve_rows if r["top1_is_gold"] == "yes")
    rerank_hits = sum(1 for r in rerank_rows if r["top1_is_gold"] == "yes")

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

    print("\n=== Retrieval Compare Summary ===")
    print(f"Cases evaluated:           {n}")
    print(f"retrieve top1 hits:        {retrieve_hits}/{n}")
    print(f"retrieve_rerank top1 hits: {rerank_hits}/{n}")
    print(f"repaired cases:            {repaired}")
    print(f"worsened cases:            {worsened}")
    print(f"unchanged good cases:      {unchanged_good}")
    print(f"unchanged bad cases:       {unchanged_bad}")

    # optional breakdown by question_type
    retrieve_by_type: dict[str, list[dict[str, Any]]] = {}
    rerank_by_type: dict[str, list[dict[str, Any]]] = {}

    for r in retrieve_rows:
        qt = r.get("question_type", "") or "(untyped)"
        retrieve_by_type.setdefault(qt, []).append(r)

    for r in rerank_rows:
        qt = r.get("question_type", "") or "(untyped)"
        rerank_by_type.setdefault(qt, []).append(r)

    if len(retrieve_by_type) > 1:
        print("\n=== Compare breakdown by question_type ===")
        for qt in sorted(retrieve_by_type.keys()):
            g0 = retrieve_by_type.get(qt, [])
            g1 = rerank_by_type.get(qt, [])
            h0 = sum(1 for r in g0 if r["top1_is_gold"] == "yes")
            h1 = sum(1 for r in g1 if r["top1_is_gold"] == "yes")
            print(f"{qt}: retrieve={h0}/{len(g0)} -> retrieve_rerank={h1}/{len(g1)}")

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
        description="Run retrieval-only evaluation from a TSV question set.")
    parser.add_argument(
        "--questions",
        default="eval/data/retrieval_eval_questions.tsv",
        help="Path to TSV question set",
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "compare"],
        help="single = run one endpoint, compare = run retrieve and retrieve_rerank together",
    )
    parser.add_argument(
        "--endpoint",
        default="retrieve",
        choices=["retrieve", "retrieve_rerank"],
        help="Which retrieval endpoint to evaluate in single mode",
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
        help="top_k passed to the endpoint",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output TSV path (optional)",
    )
    parser.add_argument(
        "--qid",
        default=None,
        help="Run only one specific qid, e.g. --qid Q04",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help="Run only questions from one document, e.g. --doc test_eval_notes.md",
    )
    parser.add_argument(
        "--contains",
        default=None,
        help="Run only questions whose text contains this substring, case-insensitive",
    )
    parser.add_argument(
        "--question-type",
        default=None,
        help="Run only one question_type, e.g. --question-type purpose_objective",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Run only one subset, e.g. --subset ab_core",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    questions = read_questions(questions_path)
    questions = filter_questions(
        questions,
        qid=args.qid,
        doc=args.doc,
        contains=args.contains,
        question_type=args.question_type,
        subset=args.subset,
    )

    if not questions:
        print("No questions matched the provided filters.", file=sys.stderr)
        return 1

    if args.mode == "single":
        if args.out is None:
            suffix = []
            if args.qid:
                suffix.append(f"qid_{args.qid}")
            if args.doc:
                suffix.append(f"doc_{args.doc.replace('.', '_')}")
            if args.contains:
                suffix.append(f"contains_{args.contains.replace(' ', '_')}")
            if args.question_type:
                suffix.append(f"qtype_{args.question_type}")
            if args.subset:
                suffix.append(f"subset_{args.subset}")
            suffix_str = "_" + "_".join(suffix) if suffix else ""
            out_path = Path(
                f"eval/data/results/{args.endpoint}{suffix_str}_eval_results.tsv")
        else:
            out_path = Path(args.out)

        try:
            rows_out = evaluate_endpoint(
                questions=questions,
                base_url=args.base_url,
                endpoint=args.endpoint,
                top_k=args.top_k,
            )
        except Exception:
            return 1

        write_tsv(rows_out, out_path)
        print_single_summary(rows_out, args.endpoint)
        print(f"\nSaved TSV to: {out_path}")
        return 0

    # compare mode
    if args.out is None:
        suffix = []
        if args.qid:
            suffix.append(f"qid_{args.qid}")
        if args.doc:
            suffix.append(f"doc_{args.doc.replace('.', '_')}")
        if args.contains:
            suffix.append(f"contains_{args.contains.replace(' ', '_')}")
        if args.question_type:
            suffix.append(f"qtype_{args.question_type}")
        if args.subset:
            suffix.append(f"subset_{args.subset}")
        suffix_str = "_" + "_".join(suffix) if suffix else ""
        out_path = Path(
            f"eval/data/results/retrieve_compare{suffix_str}_eval_results.tsv")
    else:
        out_path = Path(args.out)

    try:
        retrieve_rows = evaluate_endpoint(
            questions=questions,
            base_url=args.base_url,
            endpoint="retrieve",
            top_k=args.top_k,
        )
        rerank_rows = evaluate_endpoint(
            questions=questions,
            base_url=args.base_url,
            endpoint="retrieve_rerank",
            top_k=args.top_k,
        )
    except Exception:
        return 1

    combined_rows = retrieve_rows + rerank_rows
    write_tsv(combined_rows, out_path)
    print_compare_summary(combined_rows)
    print(f"\nSaved TSV to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
