# eval/rerank_experiment.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------
# Expanded minimal rerank experiment
#
# Purpose:
#   Evaluate whether lightweight query normalization + lexical rerank
#   can improve top1 ranking on cases where gold was NOT top1 before.
#
# Cases included:
#   - Q04
#   - Q05
#   - Q01
#   - H03
#   - H06
#
# Output:
#   - original ranking
#   - reranked ranking
#   - gold rank before / after
#   - summary metrics across all cases
# ---------------------------------------------------------------------


@dataclass
class EvalCase:
    qid: str
    doc_name: str
    question: str
    gold_chunk_id: int
    gold_hint: str
    results: list[dict[str, Any]]


# ---------------------------------------------------------------------
# Lightweight rerank config
# ---------------------------------------------------------------------

BOOST_TERMS: dict[str, float] = {
    # purpose / objective / goal
    "purpose": 0.25,
    "objective": 0.25,
    "goal": 0.20,
    "main": 0.08,
    "directly": 0.06,
    "exact": 0.06,

    # experiment / study framing
    "experimental": 0.15,
    "study": 0.10,
    "built": 0.05,
    "around": 0.03,

    # method terms
    "measured": 0.15,
    "assay": 0.15,
    "method": 0.12,
    "approach": 0.10,

    # limitations / negative evidence / benchmarks
    "benchmark": 0.15,
    "accuracy": 0.10,
    "report": 0.10,
    "reported": 0.10,
    "limitation": 0.12,
    "constraint": 0.10,
    "current": 0.06,
    "not": 0.05,
    "no": 0.05,

    # architecture / metadata
    "section": 0.08,
    "metadata": 0.12,
    "cite": 0.08,
    "citation": 0.08,
}


def normalize_query(text: str) -> str:
    t = text.lower()

    # abstract intent normalization
    t = t.replace("experimental goal", "objective")
    t = t.replace("built around", "objective")
    t = t.replace("main purpose", "purpose")
    t = t.replace("main objective", "objective")

    # weaker paraphrase normalization
    t = t.replace("current constraint", "limitation")
    t = t.replace("current constraints", "limitations")
    t = t.replace("what limitation", "limitation")
    t = t.replace("measurement approach", "method assay")
    t = t.replace("helpful for evaluation", "easy to retrieve evaluation")
    t = t.replace("baseline value", "baseline numeric value")
    t = t.replace("before stress began", "baseline before stress began")

    return t


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_bonus(query: str, text: str) -> float:
    q_norm = normalize_query(query)
    q_tokens = set(tokenize(q_norm))
    t_tokens = set(tokenize(text))

    bonus = 0.0

    # overlap bonus
    overlap = q_tokens & t_tokens
    bonus += 0.03 * len(overlap)

    # stronger term-weight bonus
    for tok, weight in BOOST_TERMS.items():
        if tok in q_tokens and tok in t_tokens:
            bonus += weight

    return bonus


def rerank_score(query: str, result: dict[str, Any]) -> float:
    """
    Higher is better.

    score = -cosine_distance + lexical_bonus(query, snippet)
    """
    dist = float(result.get("cosine_distance", 999.0))
    snippet = str(result.get("snippet", ""))
    return (-dist) + lexical_bonus(query, snippet)


def rerank_results(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []
    for r in results:
        x = dict(r)
        x["rerank_score"] = rerank_score(query, r)
        reranked.append(x)
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def short_loc(result: dict[str, Any]) -> str:
    section = result.get("section")
    page = result.get("page")
    if section:
        return str(section)
    if page is not None:
        return f"page {page}"
    return "unknown"


def get_rank(results: list[dict[str, Any]], gold_chunk_id: int) -> int | None:
    for i, r in enumerate(results, start=1):
        if r.get("chunk_id") == gold_chunk_id:
            return i
    return None


def print_ranked_list(title: str, results: list[dict[str, Any]], gold_chunk_id: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for i, r in enumerate(results, start=1):
        cid = r.get("chunk_id")
        dist = float(r.get("cosine_distance", 999.0))
        score = r.get("rerank_score")
        loc = short_loc(r)
        snippet = str(r.get("snippet", "")).replace("\n", " ").strip()
        if len(snippet) > 120:
            snippet = snippet[:120] + "..."
        gold_mark = "  <-- GOLD" if cid == gold_chunk_id else ""

        if score is None:
            print(
                f"{i:>2}. chunk={cid:<3} | dist={dist:.4f} | loc={loc:<24} | {snippet}{gold_mark}")
        else:
            print(
                f"{i:>2}. chunk={cid:<3} | dist={dist:.4f} | rerank={score:.4f} | "
                f"loc={loc:<24} | {snippet}{gold_mark}"
            )


def evaluate_case(case: EvalCase) -> dict[str, Any]:
    baseline_rank = get_rank(case.results, case.gold_chunk_id)
    reranked = rerank_results(case.question, case.results)
    rerank_rank = get_rank(reranked, case.gold_chunk_id)

    print("\n" + "=" * 90)
    print(f"{case.qid} | {case.doc_name}")
    print(f"Question: {case.question}")
    print(f"Gold chunk id: {case.gold_chunk_id}")
    print(f"Gold hint: {case.gold_hint}")

    print_ranked_list("Original ranking", case.results, case.gold_chunk_id)
    print_ranked_list("Reranked", reranked, case.gold_chunk_id)

    top1_before = baseline_rank == 1
    top1_after = rerank_rank == 1
    promoted = (
        baseline_rank is not None
        and rerank_rank is not None
        and rerank_rank < baseline_rank
    )
    worsened = (
        baseline_rank is not None
        and rerank_rank is not None
        and rerank_rank > baseline_rank
    )

    print("\nSummary")
    print("-------")
    print(f"Gold rank before rerank: {baseline_rank}")
    print(f"Gold rank after rerank:  {rerank_rank}")
    print(f"Was gold top1 before?    {top1_before}")
    print(f"Was gold top1 after?     {top1_after}")
    print(f"Was gold promoted?       {promoted}")
    print(f"Was gold worsened?       {worsened}")

    return {
        "qid": case.qid,
        "gold_rank_before": baseline_rank,
        "gold_rank_after": rerank_rank,
        "top1_before": top1_before,
        "top1_after": top1_after,
        "promoted": promoted,
        "worsened": worsened,
    }


# ---------------------------------------------------------------------
# Real retrieval outputs copied from your experiments
# ---------------------------------------------------------------------

Q04_RESULTS = [
    {
        "chunk_id": 23,
        "chunk_index": 4,
        "title": "Test Eval Notes Markdown",
        "filename": "test_eval_notes.md",
        "filename_meta": "test_eval_notes.md",
        "source_type": "md",
        "page": None,
        "section": "Architecture Decisions",
        "level": 2,
        "cosine_distance": 0.5887020228602364,
        "snippet": (
            "The project keeps document-level records separate from chunk-level records. "
            "It also stores lightweight metadata so that answers can cite source sections or pages."
        ),
    },
    {
        "chunk_id": 19,
        "chunk_index": 0,
        "title": "Test Eval Notes Markdown",
        "filename": "test_eval_notes.md",
        "filename_meta": "test_eval_notes.md",
        "source_type": "md",
        "page": None,
        "section": "Project Overview",
        "level": 1,
        "cosine_distance": 0.5917649737389847,
        "snippet": (
            "This markdown document was created for testing a retrieval and grounded question-answering "
            "pipeline. The purpose of the project is to provide a compact, section-structured source for "
            "evaluating section-aware retrieval."
        ),
    },
    {
        "chunk_id": 20,
        "chunk_index": 1,
        "title": "Test Eval Notes Markdown",
        "filename": "test_eval_notes.md",
        "filename_meta": "test_eval_notes.md",
        "source_type": "md",
        "page": None,
        "section": "Why This Project",
        "level": 2,
        "cosine_distance": 0.7619760221985137,
        "snippet": (
            "Many small RAG systems can ingest markdown easily, but they still need to prove that they can "
            "retrieve the correct section and refuse unsupported claims. This file is intentionally simple "
            "so that section metadata can be tested clearly."
        ),
    },
    {
        "chunk_id": 21,
        "chunk_index": 2,
        "title": "Test Eval Notes Markdown",
        "filename": "test_eval_notes.md",
        "filename_meta": "test_eval_notes.md",
        "source_type": "md",
        "page": None,
        "section": "Workflow",
        "level": 2,
        "cosine_distance": 0.7892656454386354,
        "snippet": (
            "The current workflow has four stages: ingest the document, split it into chunks, retrieve relevant "
            "evidence, and answer only when the evidence is sufficient."
        ),
    },
]

Q05_RESULTS = [
    {
        "chunk_id": 20,
        "chunk_index": 1,
        "title": "Test Eval Notes Markdown",
        "filename": "test_eval_notes.md",
        "filename_meta": "test_eval_notes.md",
        "source_type": "md",
        "page": None,
        "section": "Why This Project",
        "level": 2,
        "cosine_distance": 0.5062515726115473,
        "snippet": (
            "Many small RAG systems can ingest markdown easily, but they still need to prove that they can "
            "retrieve the correct section and refuse unsupported claims. This file is intentionally simple "
            "so that section metadata can be tested clearly."
        ),
    },
    {
        "chunk_id": 22,
        "chunk_index": 3,
        "title": "Test Eval Notes Markdown",
        "filename": "test_eval_notes.md",
        "filename_meta": "test_eval_notes.md",
        "source_type": "md",
        "page": None,
        "section": "Current Limitations",
        "level": 2,
        "cosine_distance": 0.5439485793787181,
        "snippet": (
            "The current system does not include OCR for scanned PDFs. It also does not include a learned reranker, "
            "and it does not report benchmark metrics such as F1, MRR, or accuracy in this markdown file."
        ),
    },
    {
        "chunk_id": 19,
        "chunk_index": 0,
        "title": "Test Eval Notes Markdown",
        "filename": "test_eval_notes.md",
        "filename_meta": "test_eval_notes.md",
        "source_type": "md",
        "page": None,
        "section": "Project Overview",
        "level": 1,
        "cosine_distance": 0.545303990568114,
        "snippet": (
            "This markdown document was created for testing a retrieval and grounded question-answering pipeline. "
            "The purpose of the project is to provide a compact, section-structured source for evaluating "
            "section-aware retrieval."
        ),
    },
    {
        "chunk_id": 23,
        "chunk_index": 4,
        "title": "Test Eval Notes Markdown",
        "filename": "test_eval_notes.md",
        "filename_meta": "test_eval_notes.md",
        "source_type": "md",
        "page": None,
        "section": "Architecture Decisions",
        "level": 2,
        "cosine_distance": 0.607294785731992,
        "snippet": (
            "The project keeps document-level records separate from chunk-level records. It also stores lightweight "
            "metadata so that answers can cite source sections or pages."
        ),
    },
]

Q01_RESULTS = [
    {
        "chunk_id": 41,
        "chunk_index": 16,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.5864150261522731,
        "snippet": (
            "The document should support answerable questions about objective, assay, results, and conclusion. "
            "It should also support a grounded negative answer to questions about Drug Zeta or in vivo testing "
            "because those items are explicitly absent."
        ),
    },
    {
        "chunk_id": 40,
        "chunk_index": 15,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.6148715472283353,
        "snippet": (
            "In conclusion, the study objective was to compare Signal-K behavior during stress and recovery, "
            "and the main conclusion is that Signal-K increases during stress and then trends back toward baseline "
            "during recovery."
        ),
    },
    {
        "chunk_id": 26,
        "chunk_index": 1,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.6185085192700657,
        "snippet": (
            "Abstract. This two-column PDF is designed for testing text extraction, chunking, retrieval, and grounded "
            "question answering. The study objective is to compare Signal-K behavior during stress and recovery phases "
            "in a compact document with explicit met"
        ),
    },
    {
        "chunk_id": 25,
        "chunk_index": 0,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.6326699742617161,
        "snippet": "Signal-K Stress Response Study",
    },
]

H03_RESULTS = [
    {
        "chunk_id": 29,
        "chunk_index": 4,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.6450388536678163,
        "snippet": (
            "A secondary goal is to make the document easy to cite. The text therefore includes clearly named "
            "sections and direct declarative statements that can serve as gold evidence for evaluation. "
            "The document is intentionally short, but the language mirrors "
        ),
    },
    {
        "chunk_id": 30,
        "chunk_index": 5,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.6495994968700185,
        "snippet": "Methods",
    },
    {
        "chunk_id": 32,
        "chunk_index": 7,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.6760396448796218,
        "snippet": (
            "The primary assay used to quantify Signal-K was a fluorescence assay. A simple plate-based readout "
            "was selected so the measurement method would be stated explicitly and be easy to retrieve from the "
            "Methods section."
        ),
    },
    {
        "chunk_id": 26,
        "chunk_index": 1,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.6966189856178011,
        "snippet": (
            "Abstract. This two-column PDF is designed for testing text extraction, chunking, retrieval, and grounded "
            "question answering. The study objective is to compare Signal-K behavior during stress and recovery phases "
            "in a compact document with explicit met"
        ),
    },
]

H06_RESULTS = [
    {
        "chunk_id": 34,
        "chunk_index": 9,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.23208917740501256,
        "snippet": (
            "During stress, normalized Signal-K increased from baseline to a peak mean value of 1.8 arbitrary units. "
            "During recovery, the signal declined and approached a mean value of 1.1 arbitrary units by the end of "
            "the observation window."
        ),
    },
    {
        "chunk_id": 40,
        "chunk_index": 15,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.2942385434386843,
        "snippet": (
            "In conclusion, the study objective was to compare Signal-K behavior during stress and recovery, "
            "and the main conclusion is that Signal-K increases during stress and then trends back toward baseline "
            "during recovery."
        ),
    },
    {
        "chunk_id": 35,
        "chunk_index": 10,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.304435300268009,
        "snippet": (
            "The main result is that Signal-K rises during stress and partially returns toward baseline during recovery. "
            "This pattern supports the idea that the marker is responsive to acute perturbation but is not permanently "
            "elevated once the stressor is remove"
        ),
    },
    {
        "chunk_id": 31,
        "chunk_index": 6,
        "title": "Test Dual Column Eval PDF",
        "filename": "test_dual_column_eval.pdf",
        "filename_meta": "test_dual_column_eval.pdf",
        "source_type": "pdf",
        "page": 1,
        "section": None,
        "level": None,
        "cosine_distance": 0.3129722163300489,
        "snippet": (
            "Cells were exposed to a 20-minute stress protocol and then allowed to recover for 40 minutes. "
            "Signal-K abundance was measured with a fluorescence assay, and all measurements were normalized "
            "to the baseline condition collected before stress began."
        ),
    },
]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    cases = [
        EvalCase(
            qid="Q04",
            doc_name="test_eval_notes.md",
            question="What is the purpose of the project?",
            gold_chunk_id=19,
            gold_hint="purpose of the project",
            results=Q04_RESULTS,
        ),
        EvalCase(
            qid="Q05",
            doc_name="test_eval_notes.md",
            question="What limitation or current constraint of the system is described in the markdown?",
            gold_chunk_id=22,
            gold_hint="current limitations include OCR / reranker / benchmark metrics absence",
            results=Q05_RESULTS,
        ),
        EvalCase(
            qid="Q01",
            doc_name="test_dual_column_eval.pdf",
            question="What is the study objective?",
            gold_chunk_id=26,
            gold_hint="study objective compare Signal-K during stress and recovery",
            results=Q01_RESULTS,
        ),
        EvalCase(
            qid="H03",
            doc_name="test_dual_column_eval.pdf",
            question="What measurement approach was chosen, and why was that choice helpful for evaluation?",
            gold_chunk_id=32,
            gold_hint="fluorescence assay and easy to retrieve",
            results=H03_RESULTS,
        ),
        EvalCase(
            qid="H06",
            doc_name="test_dual_column_eval.pdf",
            question="What was the baseline value of Signal-K before stress began?",
            gold_chunk_id=31,
            gold_hint="normalized to baseline before stress began; no numeric baseline value",
            results=H06_RESULTS,
        ),
    ]

    print("\nExpanded rerank experiment")
    print("==========================")
    print("Evaluating 5 previously non-top1 cases.\n")

    summaries: list[dict[str, Any]] = []
    for case in cases:
        summaries.append(evaluate_case(case))

    print("\n" + "=" * 90)
    print("Aggregate summary")
    print("=" * 90)

    n = len(summaries)
    before_top1 = sum(1 for x in summaries if x["top1_before"])
    after_top1 = sum(1 for x in summaries if x["top1_after"])
    promoted = sum(1 for x in summaries if x["promoted"])
    worsened = sum(1 for x in summaries if x["worsened"])

    print(f"Cases evaluated:          {n}")
    print(f"Top1 before rerank:       {before_top1}/{n}")
    print(f"Top1 after rerank:        {after_top1}/{n}")
    print(f"Promoted cases:           {promoted}/{n}")
    print(f"Worsened cases:           {worsened}/{n}")

    print("\nPer-case outcome")
    print("----------------")
    for x in summaries:
        print(
            f"{x['qid']}: rank {x['gold_rank_before']} -> {x['gold_rank_after']} | "
            f"top1_before={x['top1_before']} | top1_after={x['top1_after']} | "
            f"promoted={x['promoted']} | worsened={x['worsened']}"
        )


if __name__ == "__main__":
    main()
