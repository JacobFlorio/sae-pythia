"""Cross-scale feature matching via decoder cosine similarity.

For a given layer, loads two checkpoints (e.g. 5M and 50M token runs) and
computes the cosine similarity between every pair of decoder directions.

Since W_dec rows are already unit-norm (enforced by post_step()), the full
similarity matrix is just W_dec_A @ W_dec_B.T — no extra normalization needed.

For each feature in checkpoint A, we report:
  - its best-match feature index in checkpoint B
  - the cosine similarity of that match
  - whether the feature appears in a provided dashboard (so we can name it)

The distribution of best-match cosines tells us how much of the 5M feature
space "survives" into 50M: a sharp mass near 1.0 means stable features; a
flat distribution near 0 means the model reorganized completely.

Usage:
  uv run python scripts/match_features.py \\
    --ckpt-a checkpoints/sae_L6_d16384_k32_5M.pt \\
    --ckpt-b checkpoints/sae_L6_d16384_k32.pt \\
    --dashboard-a dashboards/features_L6.json \\
    --dashboard-b dashboards/features_L6_50M.json \\
    --output dashboards/feature_match_L6.json \\
    --top-cos 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# Checkpoint loading (mirrors dashboard.py / analyze_geometry.py)
# --------------------------------------------------------------------------- #

def load_decoder(checkpoint_path: str, device: str = "cpu") -> torch.Tensor:
    """Return W_dec (d_sae, d_model) from a checkpoint, unit-norm rows."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    W = state["W_dec"].float()
    # Re-normalize just in case (should already be unit-norm).
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return W


def load_dashboard_index(path: str | None) -> dict[int, list[dict]]:
    """Return {latent_int: [examples]} or empty dict if path is None."""
    if path is None:
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def compute_matches(
    W_a: torch.Tensor,
    W_b: torch.Tensor,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (best_idx, best_cos) for each row of W_a against all rows of W_b.

    Chunked to avoid OOM on large feature spaces (32k × 32k = 1B floats).
    """
    n_a = W_a.shape[0]
    best_cos = torch.full((n_a,), -1.0)
    best_idx = torch.zeros(n_a, dtype=torch.long)

    for start in range(0, n_a, batch_size):
        end = min(start + batch_size, n_a)
        chunk = W_a[start:end]          # (batch, d_model)
        sims = chunk @ W_b.T            # (batch, d_sae_b)
        chunk_best_cos, chunk_best_idx = sims.max(dim=1)
        best_cos[start:end] = chunk_best_cos
        best_idx[start:end] = chunk_best_idx

    return best_idx, best_cos


def _peak_activation(examples: list[dict]) -> float:
    return max((ex["activation"] for ex in examples), default=0.0)


def _top_token(examples: list[dict]) -> str:
    """Return the highlight token from the highest-activation example."""
    if not examples:
        return ""
    best = max(examples, key=lambda ex: ex["activation"])
    return best.get("highlight_token", "")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(
    W_a: torch.Tensor,
    W_b: torch.Tensor,
    dash_a: dict[int, list[dict]],
    dash_b: dict[int, list[dict]],
    top_cos_threshold: float,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    best_idx, best_cos = compute_matches(W_a, W_b)
    best_idx_np = best_idx.numpy()
    best_cos_np = best_cos.numpy()

    # Distribution stats
    thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
    distribution = {
        f"frac_cos_gt_{int(t*10):02d}": float((best_cos > t).float().mean())
        for t in thresholds
    }
    distribution["mean_best_cos"] = float(best_cos.mean())
    distribution["median_best_cos"] = float(best_cos.median())

    # Per-feature matches at or above threshold — only for features in dashboard A
    # (or all features if no dashboard supplied)
    if dash_a:
        candidates = sorted(dash_a.keys())
    else:
        candidates = list(range(W_a.shape[0]))

    matches = []
    for lat_a in candidates:
        cos_val = float(best_cos_np[lat_a])
        if cos_val < top_cos_threshold:
            continue
        lat_b = int(best_idx_np[lat_a])
        exs_a = dash_a.get(lat_a, [])
        exs_b = dash_b.get(lat_b, [])
        matches.append({
            f"latent_{label_a}": lat_a,
            f"latent_{label_b}": lat_b,
            "cosine_similarity": round(cos_val, 4),
            f"peak_activation_{label_a}": round(_peak_activation(exs_a), 2),
            f"peak_activation_{label_b}": round(_peak_activation(exs_b), 2),
            f"top_token_{label_a}": _top_token(exs_a),
            f"top_token_{label_b}": _top_token(exs_b),
            f"in_dashboard_{label_b}": lat_b in dash_b,
        })

    matches.sort(key=lambda m: m["cosine_similarity"], reverse=True)

    # Reciprocal matches: features in dash_b that also match back to their
    # declared dash_a partner (strong evidence of identity across scales).
    if dash_a and dash_b:
        best_idx_ba, best_cos_ba = compute_matches(W_b, W_a)
        best_idx_ba_np = best_idx_ba.numpy()
        for m in matches:
            la = m[f"latent_{label_a}"]
            lb = m[f"latent_{label_b}"]
            reciprocal = int(best_idx_ba_np[lb]) == la
            m["reciprocal_match"] = reciprocal
    else:
        best_cos_ba = None

    return {
        "label_a": label_a,
        "label_b": label_b,
        "d_sae_a": W_a.shape[0],
        "d_sae_b": W_b.shape[0],
        "threshold": top_cos_threshold,
        "distribution": distribution,
        "matches": matches,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Match SAE features across two checkpoints.")
    p.add_argument("--ckpt-a", required=True, help="Earlier checkpoint (e.g. 5M run)")
    p.add_argument("--ckpt-b", required=True, help="Later checkpoint (e.g. 50M run)")
    p.add_argument("--dashboard-a", default=None, help="Dashboard JSON for checkpoint A")
    p.add_argument("--dashboard-b", default=None, help="Dashboard JSON for checkpoint B")
    p.add_argument("--label-a", default="5M", help="Label for checkpoint A in output")
    p.add_argument("--label-b", default="50M", help="Label for checkpoint B in output")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--top-cos", type=float, default=0.3, help="Cosine threshold for reporting matches (default 0.3)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print(f"Loading {args.ckpt_a} ...")
    W_a = load_decoder(args.ckpt_a, args.device)
    print(f"  W_dec shape: {tuple(W_a.shape)}")

    print(f"Loading {args.ckpt_b} ...")
    W_b = load_decoder(args.ckpt_b, args.device)
    print(f"  W_dec shape: {tuple(W_b.shape)}")

    dash_a = load_dashboard_index(args.dashboard_a)
    dash_b = load_dashboard_index(args.dashboard_b)
    if dash_a:
        print(f"Dashboard A: {len(dash_a)} features")
    if dash_b:
        print(f"Dashboard B: {len(dash_b)} features")

    print("Computing matches (chunked) ...")
    report = build_report(
        W_a, W_b, dash_a, dash_b,
        top_cos_threshold=args.top_cos,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    dist = report["distribution"]
    print(f"\nBest-match cosine distribution (A→B):")
    print(f"  mean={dist['mean_best_cos']:.4f}  median={dist['median_best_cos']:.4f}")
    for k, v in dist.items():
        if k.startswith("frac_cos"):
            thresh_val = int(k.split("_gt_")[1]) / 10
            print(f"  cos > {thresh_val:.1f}: {v:.3f} ({v*W_a.shape[0]:.0f} features)")

    matches = report["matches"]
    print(f"\nMatches at cos > {args.top_cos}: {len(matches)}")
    reciprocal = [m for m in matches if m.get("reciprocal_match")]
    if reciprocal:
        print(f"  Reciprocal matches: {len(reciprocal)}")
        print(f"\n  Top reciprocal matches:")
        for m in reciprocal[:15]:
            la = m[f"latent_{args.label_a}"]
            lb = m[f"latent_{args.label_b}"]
            cos = m["cosine_similarity"]
            tok_a = m.get(f"top_token_{args.label_a}", "?")
            tok_b = m.get(f"top_token_{args.label_b}", "?")
            peak_a = m.get(f"peak_activation_{args.label_a}", 0)
            peak_b = m.get(f"peak_activation_{args.label_b}", 0)
            print(f"    {args.label_a}:{la:5d} ↔ {args.label_b}:{lb:5d}  cos={cos:.3f}  "
                  f"peak {peak_a:.1f}→{peak_b:.1f}  tok: {repr(tok_a)}→{repr(tok_b)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
