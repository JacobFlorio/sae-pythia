# sae-pythia

Top-k Sparse Autoencoders trained on the residual stream of [Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m), with tooling to surface and score interpretable features.

From-scratch reimplementation of the top-k SAE architecture from Gao et al. 2024 ([*Scaling and evaluating sparse autoencoders*](https://arxiv.org/abs/2406.04093)), applied across layers 3, 6, and 9 of a small open model. The full pipeline fits on a single consumer GPU.

## Key findings

| | Layer 3 | Layer 6 | Layer 9 |
|---|---|---|---|
| FVU @ 5M tokens | 0.078 | **0.074** | 0.125 |
| FVU @ 50M tokens | **0.042** | 0.057 | 0.099 |
| Unique feature clusters @ 5M | 3 | 5 | 3 |
| Unique feature clusters @ 50M | 27 | 26 | **39** |
| Auto-interp balanced accuracy @ 50M | 0.901 | 0.804 | 0.858 |

**A phase transition occurs between 5M and 50M tokens.** Dead latents (41% at 5M) collapse to near-zero at 50M, and unique feature clusters jump from 3–5 to 26–39. Layer 9 flips from worst feature diversity to best.

**Features are stable across training scales.** Auto-insurance boilerplate (L6), greatest-common-divisor problems (L9), and biomedical citation markup (L6) all appear at both 5M and 50M tokens with higher activation at the longer run — confirming they are real learned features, not noise.

**The layer 9 FVU/diversity paradox.** Layer 9 has the richest feature set at 50M (39 unique clusters, 25/50 features purely mid-document) but the worst reconstruction (FVU 0.099). Decoder geometry analysis rules out superposition as the cause — the difficulty is intrinsic to the deeper residual stream.

Full analysis: [FINDINGS.md](FINDINGS.md)

### Visualizations

| FVU by layer and scale | Feature diversity phase transition | Auto-interp scores |
|---|---|---|
| ![FVU](assets/fvu_comparison.svg) | ![Diversity](assets/feature_diversity.svg) | ![Autointerp](assets/autointerp_scores.svg) |

## Why top-k?

Classic L1-penalty SAEs trade reconstruction quality against sparsity via a Lagrange multiplier that is notoriously hard to tune. Top-k SAEs sidestep this: they keep the `k` largest pre-activations per token and zero the rest, giving exact control over the active-feature count and eliminating shrinkage bias on the surviving features. The tradeoff is that dead latents can accumulate, so we include an auxiliary "AuxK" loss that forces the top dead latents to reconstruct the residual — also from Gao et al.

## What's here

Library (`src/sae/`):

- `model.py` — `TopKSAE` module: encoder, top-k activation, unit-norm decoder, AuxK dead-latent revival.
- `activations.py` — streaming activation collector that hooks a chosen layer's residual stream and yields `(B, d_model)` batches without materializing the full corpus in RAM.
- `train.py` — training loop with dead-latent tracking, FVU metric, and checkpoint saving.
- `dashboard.py` — per-feature max-activating-example extraction over streamed Pile documents.
- `autointerp.py` — two-stage Claude-based auto-interpretability: explainer generates a one-sentence description; scorer grades it by forced-choice discrimination (balanced accuracy).

CLI entry points (`scripts/`):

- `train_sae.py` — train a TopK SAE on a chosen layer.
- `build_dashboard.py` — extract top-N max-activating examples per latent from a checkpoint.
- `sample_dashboard.py` — slice a full dashboard to a committable sample; supports `peak`, `mid_document`, and `dedupe` ranking modes.
- `run_autointerp.py` — run the Claude explainer + scorer over a dashboard JSON.
- `compare_layers.py` — compute BOS-cluster size, strict mid-doc count, unique-cluster count, and peak-activation stats across multiple layer dashboards.
- `analyze_geometry.py` — measure decoder superposition via cosine similarity, uniformity loss, and effective rank from SVD.

## Hardware target

Designed for a single RTX 5080 (16 GB). Pythia-160M in fp16 + a 16k-feature SAE on `d_model=768` fits comfortably with room for a 4k-token activation buffer. 50M-token runs take ~24 min per layer.

## Quickstart

```bash
uv sync

# Train a TopK SAE on layer 6 (~24 min at 50M tokens on RTX 5080)
uv run python scripts/train_sae.py --model EleutherAI/pythia-160m --layer 6 --d-sae 16384 --k 32 --tokens 50_000_000 --batch-size 4096

# Build the feature dashboard
uv run python scripts/build_dashboard.py --checkpoint checkpoints/sae_L6_d16384_k32.pt --layer 6 --output dashboards/features_L6.json --num-docs 500

# Sample top-30 deduplicated features for inspection
uv run python scripts/sample_dashboard.py --input dashboards/features_L6.json --output dashboards/features_L6_dedupe.json --rank-by dedupe --top-n-features 30

# Auto-interp with Claude (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
uv run python scripts/run_autointerp.py --dashboard dashboards/features_L6.json --output dashboards/autointerp_L6.json --num-features 30
```

## Committed artifacts

- `dashboards/features_L*_sample_dedupe.json` — top-30 Jaccard-deduplicated features per layer (5M run)
- `dashboards/features_L*_50M_dedupe.json` — top-30 deduplicated features per layer (50M run)
- `dashboards/layer_comparison.json` / `layer_comparison_50M.json` — cluster metrics across layers
- `dashboards/geometry.json` — decoder superposition metrics
- `dashboards/autointerp_L*.json` — Claude descriptions + balanced-accuracy scores

## License

MIT — see [LICENSE](LICENSE).
