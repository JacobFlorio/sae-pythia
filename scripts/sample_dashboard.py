"""Sample the top-N most-active features from a full dashboard JSON.

Produces a small, committable slice of the dashboard so the repo can ship a
concrete example artifact without committing the full multi-megabyte file.

Two ranking modes:

- `peak` (default): sort features by their single highest activation across
  all examples. This surfaces whichever latents fire hardest overall — which,
  empirically, is dominated by "BOS-attending" latents that fire on the first
  content token of a document regardless of what that token is.

- `mid_document`: sort features by their highest activation at token
  positions >= `--min-pos` (default 5). Filters out the BOS cluster.

- `dedupe`: sort features by peak activation, then collapse duplicates by
  document fingerprint — if multiple features have top-example sets drawn
  from the same documents, keep only the single highest-activation
  representative. This is what surfaces real structural diversity.
"""

import argparse
import json


def _rank_key_peak(exs: list[dict]) -> float:
    return max((ex["activation"] for ex in exs), default=0.0)


def _rank_key_mid(exs: list[dict], min_pos: int) -> float:
    return max(
        (ex["activation"] for ex in exs if ex["token_pos"] >= min_pos),
        default=0.0,
    )


def _doc_fingerprint(exs: list[dict]) -> frozenset[int]:
    return frozenset(ex["doc_id"] for ex in exs)


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="dashboards/features.json")
    p.add_argument("--output", default="dashboards/features_sample.json")
    p.add_argument("--top-n-features", type=int, default=50)
    p.add_argument("--examples-per-feature", type=int, default=8)
    p.add_argument(
        "--rank-by",
        choices=["peak", "mid_document", "dedupe"],
        default="peak",
    )
    p.add_argument(
        "--min-pos",
        type=int,
        default=5,
        help="For rank-by=mid_document: minimum token position to count",
    )
    p.add_argument(
        "--filter-examples",
        action="store_true",
        help="Drop examples below --min-pos from the committed output (mid_document only)",
    )
    p.add_argument(
        "--dedupe-jaccard",
        type=float,
        default=0.5,
        help="For rank-by=dedupe: skip features whose doc-set overlap with any already-selected feature exceeds this threshold",
    )
    args = p.parse_args()

    with open(args.input) as f:
        full: dict[str, list[dict]] = json.load(f)

    if args.rank_by == "mid_document":
        key_fn = lambda exs: _rank_key_mid(exs, args.min_pos)  # noqa: E731
    else:
        key_fn = _rank_key_peak

    ranked = sorted(full.items(), key=lambda kv: key_fn(kv[1]), reverse=True)

    sample: dict[str, list[dict]] = {}
    seen_fingerprints: list[frozenset[int]] = []
    for lat, exs in ranked:
        if len(sample) >= args.top_n_features:
            break
        if args.rank_by == "mid_document" and args.filter_examples:
            exs = [e for e in exs if e["token_pos"] >= args.min_pos]
        if not exs:
            continue
        if args.rank_by == "dedupe":
            fp = _doc_fingerprint(exs)
            if any(_jaccard(fp, prev) > args.dedupe_jaccard for prev in seen_fingerprints):
                continue
            seen_fingerprints.append(fp)
        sample[lat] = exs[: args.examples_per_feature]

    with open(args.output, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Wrote {len(sample)} features to {args.output} (rank_by={args.rank_by})")


if __name__ == "__main__":
    main()
