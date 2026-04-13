"""Measure decoder geometry across SAE checkpoints.

Computes superposition metrics from Elhage et al. 2022 on the decoder
weight matrix W_dec (shape: d_sae × d_model, unit-norm rows by construction):

- **Mean cosine similarity** between random pairs of decoder columns:
  values near 0 = near-orthogonal (low superposition); values near 1 = high
  interference / superposition.
- **Fraction of pairs** with |cosine similarity| > threshold: a count-based
  superposition measure.
- **Uniformity loss** (Wang et al. 2020): measures how evenly the decoder
  directions are spread on the unit hypersphere. Lower = more uniform.
- **Effective rank** of W_dec via the participation ratio of singular values:
  how many directions the SAE actually uses.

Running this across layers 3, 6, and 9 tests whether the layer 9 reconstruction
paradox (more live latents, worse FVU) is explained by higher superposition /
lower effective rank in the decoder at depth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sae.model import TopKSAE, TopKSAEConfig


@torch.no_grad()
def analyze(checkpoint_path: str, n_pairs: int = 50_000, device: str = "cuda") -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = TopKSAEConfig(**ckpt["cfg"])
    sae = TopKSAE(cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()

    W = sae.W_dec.float()  # (d_sae, d_model), unit-norm rows
    d_sae = W.shape[0]

    # --- cosine similarity on random pairs ---
    idx_a = torch.randint(0, d_sae, (n_pairs,), device=device)
    idx_b = torch.randint(0, d_sae, (n_pairs,), device=device)
    # Avoid same-index pairs
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % d_sae
    cos = (W[idx_a] * W[idx_b]).sum(dim=-1)  # dot product = cosine (unit-norm)
    mean_cos = cos.mean().item()
    frac_high = (cos.abs() > 0.1).float().mean().item()

    # --- uniformity loss (log mean exp of pairwise squared distances) ---
    # Sample a subset for efficiency
    n_uni = min(d_sae, 2048)
    perm = torch.randperm(d_sae, device=device)[:n_uni]
    W_sub = W[perm]  # (n_uni, d_model)
    sq_dists = torch.cdist(W_sub, W_sub, p=2).pow(2)
    # Exclude diagonal
    mask = ~torch.eye(n_uni, dtype=torch.bool, device=device)
    uniformity = sq_dists[mask].mul(-2).exp().mean().log().item()

    # --- effective rank via participation ratio of singular values ---
    # SVD on a subset of rows if d_sae is large
    n_svd = min(d_sae, 4096)
    perm2 = torch.randperm(d_sae, device=device)[:n_svd]
    _, S, _ = torch.linalg.svd(W[perm2], full_matrices=False)
    S = S.float()
    p = (S ** 2) / (S ** 2).sum()
    effective_rank = torch.exp(-(p * (p + 1e-10).log()).sum()).item()

    # --- live latent count from steps_since_fired buffer ---
    threshold = cfg.dead_steps_threshold
    dead = int((sae.steps_since_fired > threshold).sum().item())
    live = d_sae - dead

    return {
        "checkpoint": str(checkpoint_path),
        "d_sae": d_sae,
        "d_model": cfg.d_model,
        "live_latents": live,
        "dead_latents": dead,
        "dead_fraction": round(dead / d_sae, 4),
        "mean_cosine_similarity": round(mean_cos, 6),
        "frac_pairs_cos_gt_0_1": round(frac_high, 4),
        "uniformity_loss": round(uniformity, 4),
        "effective_rank": round(effective_rank, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--n-pairs", type=int, default=50_000)
    p.add_argument("--output", default="dashboards/geometry.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    labels = args.labels or [Path(c).stem for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise SystemExit("--labels must match --checkpoints length")

    results = {}
    for label, ckpt in zip(labels, args.checkpoints, strict=True):
        print(f"Analyzing {label} …")
        results[label] = analyze(ckpt, n_pairs=args.n_pairs, device=args.device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty table
    cols = ["live_latents", "dead_fraction", "mean_cosine_similarity",
            "frac_pairs_cos_gt_0_1", "uniformity_loss", "effective_rank"]
    header = f"{'layer':<12}" + "".join(f"{c:>28}" for c in cols)
    print("\n" + header)
    for label, r in results.items():
        row = f"{label:<12}" + "".join(f"{r[c]:>28}" for c in cols)
        print(row)
    print(f"\nWrote full results to {args.output}")


if __name__ == "__main__":
    main()
