import argparse
import json
import time
import requests

API = "http://127.0.0.1:8000"


def hit_at_k(retrieved, expected):
    return 1 if any(x in retrieved for x in expected) else 0


def mrr(retrieved, expected):
    for i, x in enumerate(retrieved):
        if x in expected:
            return 1.0 / (i + 1)
    return 0.0


def pick_expected(ex):
    """
    Priority: pages > chunk_indexes > chunk_ids
    Returns (label_name, expected_set). If empty -> unanswerable.
    """
    if "expected_pages" in ex:
        return "page", set(ex["expected_pages"])
    if "expected_chunk_indexes" in ex:
        return "chunk_index", set(ex["expected_chunk_indexes"])
    if "expected_chunk_ids" in ex:
        return "chunk_id", set(ex["expected_chunk_ids"])
    raise KeyError(
        "Need one of expected_pages / expected_chunk_indexes / expected_chunk_ids")


def pick_retrieved(results, label_name):
    if label_name == "page":
        vals = [c.get("page") for c in results]
        return [v for v in vals if v is not None]
    if label_name == "chunk_index":
        return [c.get("chunk_index") for c in results]
    if label_name == "chunk_id":
        return [c.get("chunk_id") for c in results]
    raise ValueError(f"Unknown label_name: {label_name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--document_id", type=int, required=True)
    p.add_argument("--qa", type=str, required=True)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--skip_unanswerable", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    n = 0
    hit = 0
    mrr_sum = 0.0
    latencies = []
    unanswerable = 0

    with open(args.qa, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            q = ex["question"]

            label_name, expected = pick_expected(ex)

            if args.skip_unanswerable and len(expected) == 0:
                unanswerable += 1
                continue

            t0 = time.time()
            r = requests.post(
                f"{API}/retrieve",
                json={"query_text": q, "top_k": args.top_k,
                      "document_id": args.document_id},
                timeout=60,
            )
            dt = time.time() - t0
            latencies.append(dt)

            r.raise_for_status()
            data = r.json()

            retrieved = pick_retrieved(data["results"], label_name)

            h = hit_at_k(retrieved, expected)
            rr = mrr(retrieved, expected)

            hit += h
            mrr_sum += rr
            n += 1

            if args.debug:
                print("\nQ:", q)
                print("label:", label_name)
                print("expected:", sorted(list(expected)))
                print("retrieved:", retrieved)
                print("top:", [(c.get("chunk_id"), c.get("chunk_index"), c.get(
                    "page"), c.get("cosine_distance")) for c in data["results"]])

    report = {
        "n_scored": n,
        "top_k": args.top_k,
        "hit_at_k": hit / n if n else None,
        "mrr": mrr_sum / n if n else None,
        "avg_latency_s": sum(latencies) / n if n else None,
        "unanswerable_skipped": unanswerable if args.skip_unanswerable else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
