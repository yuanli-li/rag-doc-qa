import argparse
import json
import requests

API = "http://127.0.0.1:8000"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--document_id", type=int, required=True)
    p.add_argument("--qa", type=str, required=True)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--filter_thr", type=float, default=0.85)
    p.add_argument("--refuse_thresholds", type=str,
                   default="0.55,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    args = p.parse_args()

    refuse_thresholds = [float(x.strip())
                         for x in args.refuse_thresholds.split(",") if x.strip()]

    # Load QA (expects expected_pages)
    examples = []
    with open(args.qa, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if "expected_pages" not in ex:
                raise KeyError(
                    "tune_refuse_threshold.py expects expected_pages in QA jsonl")
            examples.append((ex["question"], ex["expected_pages"]))

    for refuse_thr in refuse_thresholds:
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

            if not results:
                # no retrieval at all => treat as refused
                refused = True
                kept = []
                best_dist = None
            else:
                best_dist = float(results[0]["cosine_distance"])
                refused = (best_dist > refuse_thr)
                kept = [c for c in results if float(
                    c["cosine_distance"]) <= args.filter_thr]

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
                    kept_pages = [c.get("page") for c in kept]
                    if any(p in expected_pages for p in kept_pages):
                        hit_answerable += 1

        out = {
            "filter_thr": args.filter_thr,
            "refuse_thr": refuse_thr,
            "answerable_n": n_answerable,
            "answerable_hit_rate": (hit_answerable / n_answerable) if n_answerable else None,
            "answerable_refused_rate": (refused_answerable / n_answerable) if n_answerable else None,
            "unanswerable_n": n_unanswerable,
            "unanswerable_refusal_accuracy": (refused_unanswerable / n_unanswerable) if n_unanswerable else None,
            "unanswerable_answered_rate": (answered_unanswerable / n_unanswerable) if n_unanswerable else None,
        }
        print(out)


if __name__ == "__main__":
    main()
