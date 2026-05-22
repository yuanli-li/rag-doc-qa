import argparse
import json
import time
import re
import requests

API = "http://127.0.0.1:8000"


def has_citation(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text or ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--document_id", type=int, required=True)
    p.add_argument("--qa", type=str, required=True)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--skip_unanswerable", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    n = 0
    n_answerable = 0
    n_unanswerable = 0

    has_cite = 0
    refused_answerable = 0

    refused_unanswerable = 0
    answered_unanswerable = 0

    rate_limited = 0
    latencies = []
    server_errors = 0

    with open(args.qa, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            q = ex["question"]

            # determine answerable/unanswerable
            expected_pages = ex.get("expected_pages")
            expected_chunk_indexes = ex.get("expected_chunk_indexes")
            expected_chunk_ids = ex.get("expected_chunk_ids")

            expected = None
            if expected_pages is not None:
                expected = expected_pages
            elif expected_chunk_indexes is not None:
                expected = expected_chunk_indexes
            elif expected_chunk_ids is not None:
                expected = expected_chunk_ids
            else:
                raise KeyError(
                    "Need expected_pages or expected_chunk_indexes or expected_chunk_ids")

            is_unanswerable = (len(expected) == 0)

            if args.skip_unanswerable and is_unanswerable:
                continue

            t0 = time.time()
            r = requests.post(
                f"{API}/ask",
                json={"query_text": q, "top_k": args.top_k,
                      "document_id": args.document_id},
                timeout=120,
            )
            dt = time.time() - t0
            latencies.append(dt)

            if r.status_code == 429:
                rate_limited += 1
                if args.debug:
                    print("\nRATE LIMITED on:", q)
                    print(r.text[:400])
                continue

            # --- handle non-200 responses without crashing ---
            if r.status_code != 200:
                print("\n[ERROR] /ask failed")
                print("Q:", q)
                print("HTTP:", r.status_code)
                print("Body (first 800 chars):")
                print(r.text[:800])
                # count & continue
                if r.status_code == 429:
                    rate_limited += 1
                else:
                    server_errors += 1
                continue
            data = r.json()

            answer = data.get("answer", "") or ""
            refused = bool(data.get("refused", False))

            n += 1
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
                if has_citation(answer):
                    has_cite += 1

            if args.debug:
                print("\nQ:", q)
                print("unanswerable:", is_unanswerable, "refused:", refused)
                print("answer:", answer[:200])

    report = {
        "n_scored": n,
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "has_citation_rate_answerable": (has_cite / n_answerable) if n_answerable else None,
        "refused_rate_answerable": (refused_answerable / n_answerable) if n_answerable else None,
        "refusal_accuracy_unanswerable": (refused_unanswerable / n_unanswerable) if n_unanswerable else None,
        "hallucination_rate_unanswerable": (answered_unanswerable / n_unanswerable) if n_unanswerable else None,
        "rate_limit_count": rate_limited,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "server_error_count": server_errors,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
