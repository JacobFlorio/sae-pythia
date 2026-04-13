# sae-pythia

Top-k Sparse Autoencoders trained on the residual stream of [Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m), with tooling to surface interpretable features.

This is a from-scratch reimplementation of the top-k SAE architecture from Gao et al. 2024 ([*Scaling and evaluating sparse autoencoders*](https://arxiv.org/abs/2406.04093)), applied to a small open model so the whole training loop fits on a single consumer GPU.

## Why top-k?

Classic L1-penalty SAEs trade reconstruction quality against sparsity via a Lagrange multiplier that is notoriously hard to tune. Top-k SAEs sidestep this: they keep the `k` largest pre-activations per token and zero the rest, giving exact control over the active-feature count and eliminating shrinkage bias on the surviving features. The tradeoff is that dead latents can accumulate, so we include an auxiliary "AuxK" loss that forces the top dead latents to reconstruct the residual — also from the Gao et al. paper.

## What's here

Library (`src/sae/`):

- `model.py` — `TopKSAE` module: encoder, top-k activation, unit-norm decoder, AuxK dead-latent revival.
- `activations.py` — streaming activation collector that hooks a chosen layer's residual stream from a HF transformer and yields `(B, d_model)` batches without materializing the full corpus in RAM.
- `train.py` — training loop with dead-latent tracking, reconstruction-FVU metric, and checkpoint saving.
- `dashboard.py` — per-feature max-activating-example extraction over streamed Pile documents, producing a JSON that can be fed to a viewer or notebook.
- `autointerp.py` — two-stage Claude-based auto-interpretability: an explainer generates a one-sentence description of each feature from its top-activating snippets, and a scorer grades the description by forced-choice discrimination against distractor snippets (balanced accuracy).

CLI entry points (`scripts/`):

- `train_sae.py` — train a TopK SAE on a chosen layer.
- `build_dashboard.py` — extract top-N max-activating examples per latent from a trained checkpoint.
- `sample_dashboard.py` — slice the full dashboard down to a small committable sample.
- `run_autointerp.py` — run the Claude explainer + scorer over a dashboard JSON.

## Hardware target

Designed for a single RTX 5080 (16 GB). Pythia-160M in fp16 + a 16k-feature SAE on `d_model=768` fits comfortably with room for a 4k-token activation buffer.

## Quickstart

Reproduce the smoke-test run below end-to-end:

```bash
uv sync

# 1. Train a TopK SAE on layer 6 of Pythia-160M (~2 min on an RTX 5080).
uv run python scripts/train_sae.py \
    --model EleutherAI/pythia-160m \
    --layer 6 \
    --d-sae 16384 \
    --k 32 \
    --tokens 5_000_000 \
    --batch-size 4096

# 2. Build the per-feature max-activating-example dashboard.
uv run python scripts/build_dashboard.py \
    --checkpoint checkpoints/sae_L6_d16384_k32.pt \
    --layer 6 \
    --output dashboards/features.json \
    --num-docs 500

# 3. (optional) Auto-interp the top features with Claude.
export ANTHROPIC_API_KEY=sk-ant-...
uv run python scripts/run_autointerp.py \
    --dashboard dashboards/features.json \
    --output dashboards/autointerp.json \
    --num-features 32
```

For a longer, more publication-quality run, bump `--tokens` to `50_000_000` or more — at that scale the dead-latent fraction drops substantially and features sharpen.

## Results (smoke test)

First end-to-end run, intended to validate the pipeline rather than produce publishable feature quality:

| Setting        | Value                      |
| -------------- | -------------------------- |
| Base model     | `EleutherAI/pythia-160m`   |
| Layer          | 6 (residual stream)        |
| `d_sae`        | 16,384                     |
| `k`            | 32                         |
| Training data  | `monology/pile-uncopyrighted`, streaming |
| Tokens seen    | 5,000,000                  |
| **FVU**        | **0.074** (≈93% variance reconstructed) |
| Dead latents   | 6,734 / 16,384 (~41%)      |
| Wallclock      | ~2 min on RTX 5080         |

The reconstruction quality is already solid at 5M tokens. The dead-latent fraction is high — this is the expected failure mode at short training runs, and is the main thing longer runs (and the AuxK coefficient) should improve. Two 50-feature slices of the resulting dashboard are committed:

- [`dashboards/features_sample.json`](dashboards/features_sample.json) — naive peak-activation ranking (dominated by duplicated "BOS absorption" latents; kept as evidence of the pathology).
- [`dashboards/features_sample_dedupe.json`](dashboards/features_sample_dedupe.json) — doc-fingerprint deduplicated, which actually surfaces semantic features.

A writeup of what the SAE is and isn't capturing — including concrete features it found (auto-insurance policy boilerplate, patent "Field of the Invention" headers, Apache-license warranty clauses, bibliographic citation markup) and the two pathologies (BOS super-cluster, paragraph-break duplicates) — lives in [`FINDINGS.md`](FINDINGS.md).

## Status

Work in progress. Roadmap:

- [x] Top-k SAE module + AuxK loss
- [x] Streaming activation pipeline
- [x] Training loop
- [x] Feature dashboard with max-activating examples
- [x] Automated interpretability scoring (feature → natural-language description)
- [ ] Writeup comparing feature quality across layers 3/6/9

## License

MIT — see [LICENSE](LICENSE).
