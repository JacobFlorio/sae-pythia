"""Compute cross-layer comparison metrics from per-layer dashboard JSONs.

Given dashboards produced from SAEs trained on different layers of the same
base model, emit a per-layer report covering:

- BOS-cluster size: how many of the top-K peak-activation latents fire at
  token_pos=0 with a nearly identical doc fingerprint. This is the
  duplication pathology documented in FINDINGS.md.
- Mid-document latent count: of the top-K peak-activation latents, how many
  have *any* top-example at token_pos >= `--min-pos`. A higher count means
  the SAE spends more of its live latent budget on in-document content.
- Unique-fingerprint count: number of distinct doc-fingerprints among the
  top-K latents after Jaccard-threshold deduplication. A proxy for how many
  truly distinct feature clusters the layer has.
- Peak-activation distribution: mean / median / min of the per-latent peak
  activations in the top-K. Helps see whether semantic features live in a
  different activation regime than the positional absorbers.

Reconstruction FVU and dead-latent counts come from the training run itself
and should be pasted in manually from the tqdm log; this script does not
re-run the model.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _doc_fingerprint(exs: list[dict]) -> frozenset[int]:
    return frozenset(ex["doc_id"] for ex in exs)


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _analyze(dashboard_path: Path, top_k: int, min_pos: int, jaccard: float) -> dict:
    with open(dashboard_path) as f:
        full: dict[str, list[dict]] = json.load(f)

    ranked = sorted(
        full.items(),
        key=lambda kv: max((ex["activation"] for ex in kv[1]), default=0.0),
        reverse=True,
    )
    top = ranked[:top_k]

    bos_count = sum(
        1 for _, exs in top if max((ex["token_pos"] for ex in exs), default=0) == 0
    )
    strict_mid_count = sum(
        1 for _, exs in top if all(ex["token_pos"] >= min_pos for ex in exs)
    )
    any_mid_count = sum(
        1 for _, exs in top if any(ex["token_pos"] >= min_pos for ex in exs)
    )

    unique_clusters: list[frozenset[int]] = []
    for _, exs in top:
        fp = _doc_fingerprint(exs)
        if any(_jaccard(fp, prev) > jaccard for prev in unique_clusters):
            continue
        unique_clusters.append(fp)

    peaks = [max(ex["activation"] for ex in exs) for _, exs in top]
    return {
        "dashboard": str(dashboard_path),
        "total_features_in_dashboard": len(full),
        "top_k": top_k,
        "bos_cluster_size": bos_count,
        "strict_mid_document_count": strict_mid_count,
        "any_mid_document_count": any_mid_count,
        "unique_clusters_after_dedupe": len(unique_clusters),
        "peak_activation_mean": round(statistics.mean(peaks), 2),
        "peak_activation_median": round(statistics.median(peaks), 2),
        "peak_activation_min": round(min(peaks), 2),
        "peak_activation_max": round(max(peaks), 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dashboards",
        nargs="+",
        required=True,
        help="One or more dashboard JSONs, one per layer (e.g. layer3.json layer6.json layer9.json)",
    )
    p.add_argument("--labels", nargs="+", help="Optional labels, one per dashboard")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-pos", type=int, default=5)
    p.add_argument("--jaccard", type=float, default=0.5)
    p.add_argument("--output", default="dashboards/layer_comparison.json")
    args = p.parse_args()

    labels = args.labels or [Path(d).stem for d in args.dashboards]
    if len(labels) != len(args.dashboards):
        raise SystemExit("--labels must have same length as --dashboards")

    report = {}
    for label, path in zip(labels, args.dashboards, strict=True):
        report[label] = _analyze(Path(path), args.top_k, args.min_pos, args.jaccard)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"{'layer':<12} {'bos':>5} {'strict_mid':>11} {'clust':>6} {'peak_med':>10}")
    for label, r in report.items():
        print(
            f"{label:<12} {r['bos_cluster_size']:>5} {r['strict_mid_document_count']:>11} "
            f"{r['unique_clusters_after_dedupe']:>6} {r['peak_activation_median']:>10.1f}"
        )
    print(f"\nWrote full report to {args.output}")


if __name__ == "__main__":
    main()
