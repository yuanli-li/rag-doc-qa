import argparse
import json
import time
import requests

API = "http://127.0.0.1:8000"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--document_id", type=int, required=True)
    p.add_argument("--qa", type=str, required=True)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--thresholds", type=str,
                   default="0.45,0.55,0.65,0.75,0.85,0.95")
    args = p.parse_args()

    thresholds = [float(x.strip())
                  for x in args.thresholds.split(",") if x.strip()]

    # Load QA
    examples = []
    with open(args.qa, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                # Determine answerable by expected_pages (recommended)
                expected_pages = ex.get("expected_pages")
                if expected_pages is None:
                    raise KeyError(
                        "QA must contain expected_pages for tune_threshold.py")
                examples.append((ex["question"], expected_pages))

    # We only use /retrieve (no LLM) to tune retrieval threshold behavior.
    # For each question, we get top_k retrieval results with cosine_distance and page.
    # Then simulate: keep chunks with dist <= thr, and "refuse" if kept is empty.
    for thr in thresholds:
        n_answerable = 0
        n_unanswerable = 0

        hit_answerable = 0
        refused_answerable = 0

        refused_unanswerable = 0
        answered_unanswerable = 0

        for (q, expected_pages) in examples:
            is_unanswerable = (len(expected_pages) == 0)

            r = requests.post(
                f"{API}/retrieve",
                json={"query_text": q, "top_k": args.top_k,
                      "document_id": args.document_id},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])

            kept = [c for c in results if c.get(
                "cosine_distance") is not None and float(c["cosine_distance"]) <= thr]
            refused = (len(kept) == 0)

            if is_unanswerable:
                n_unanswerable += 1
                if refused:
                    refused_unanswerable += 1
                else:
                    answered_unanswerable += 1
            else:
                n_answerable += 1
                if refused:
                    refused_answerable += 1
                else:
                    # check if any kept result has expected page
                    kept_pages = [c.get("page") for c in kept]
                    if any(p in expected_pages for p in kept_pages):
                        hit_answerable += 1

        print({
            "thr": thr,
            "answerable_n": n_answerable,
            "answerable_hit_rate": (hit_answerable / n_answerable) if n_answerable else None,
            "answerable_refused_rate": (refused_answerable / n_answerable) if n_answerable else None,
            "unanswerable_n": n_unanswerable,
            "unanswerable_refusal_accuracy": (refused_unanswerable / n_unanswerable) if n_unanswerable else None,
            "unanswerable_answered_rate": (answered_unanswerable / n_unanswerable) if n_unanswerable else None,
        })


if __name__ == "__main__":
    main()
