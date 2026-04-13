"""Sample the top-N most-active features from a full dashboard JSON.

Produces a small, committable slice of the dashboard so the repo can ship a
concrete example artifact without committing the full multi-megabyte file.
Ranking is by the peak activation observed for each latent.
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="dashboards/features.json")
    p.add_argument("--output", default="dashboards/features_sample.json")
    p.add_argument("--top-n-features", type=int, default=50)
    p.add_argument("--examples-per-feature", type=int, default=8)
    args = p.parse_args()

    with open(args.input) as f:
        full: dict[str, list[dict]] = json.load(f)

    ranked = sorted(
        full.items(),
        key=lambda kv: max((ex["activation"] for ex in kv[1]), default=0.0),
        reverse=True,
    )
    sample = {
        lat: exs[: args.examples_per_feature]
        for lat, exs in ranked[: args.top_n_features]
    }

    with open(args.output, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Wrote {len(sample)} features to {args.output}")


if __name__ == "__main__":
    main()
