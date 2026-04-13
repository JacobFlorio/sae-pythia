# sae-pythia

Top-k Sparse Autoencoders trained on the residual stream of [Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m), with tooling to surface interpretable features.

This is a from-scratch reimplementation of the top-k SAE architecture from Gao et al. 2024 ([*Scaling and evaluating sparse autoencoders*](https://arxiv.org/abs/2406.04093)), applied to a small open model so the whole training loop fits on a single consumer GPU.

## Why top-k?

Classic L1-penalty SAEs trade reconstruction quality against sparsity via a Lagrange multiplier that is notoriously hard to tune. Top-k SAEs sidestep this: they keep the `k` largest pre-activations per token and zero the rest, giving exact control over the active-feature count and eliminating shrinkage bias on the surviving features. The tradeoff is that dead latents can accumulate, so we include an auxiliary "AuxK" loss that forces the top dead latents to reconstruct the residual — also from the Gao et al. paper.

## What's here

- `src/sae/model.py` — `TopKSAE` module (encoder, top-k activation, unit-norm decoder, AuxK dead-latent revival).
- `src/sae/activations.py` — streaming activation collector that hooks a chosen layer's residual stream from a HF transformer and yields `(B, d_model)` batches without materializing the full corpus in RAM.
- `src/sae/train.py` — training loop with dead-latent tracking, reconstruction-FVU metric, and checkpoint saving.
- `src/sae/dashboard.py` — per-feature max-activating-example extraction for inspection.
- `scripts/train_sae.py` — CLI entry point.

## Hardware target

Designed for a single RTX 5080 (16 GB). Pythia-160M in fp16 + a 16k-feature SAE on `d_model=768` fits comfortably with room for a 4k-token activation buffer.

## Quickstart

```bash
uv sync
uv run python scripts/train_sae.py \
    --model EleutherAI/pythia-160m \
    --layer 6 \
    --d-sae 16384 \
    --k 32 \
    --tokens 50_000_000
```

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

The reconstruction quality is already solid at 5M tokens. The dead-latent fraction is high — this is the expected failure mode at short training runs, and is the main thing longer runs (and the AuxK coefficient) should improve. A 50-feature slice of the resulting dashboard is committed at [`dashboards/features_sample.json`](dashboards/features_sample.json); the full 16k-feature dashboard is gitignored.

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
