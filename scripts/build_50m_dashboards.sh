#!/usr/bin/env bash
set -e
uv run python scripts/build_dashboard.py --checkpoint checkpoints/sae_L3_d16384_k32_50M.pt --layer 3 --output dashboards/features_L3_50M.json --num-docs 500
uv run python scripts/build_dashboard.py --checkpoint checkpoints/sae_L6_d16384_k32.pt --layer 6 --output dashboards/features_L6_50M.json --num-docs 500
uv run python scripts/build_dashboard.py --checkpoint checkpoints/sae_L9_d16384_k32.pt --layer 9 --output dashboards/features_L9_50M.json --num-docs 500
uv run python scripts/compare_layers.py --dashboards dashboards/features_L3_50M.json dashboards/features_L6_50M.json dashboards/features_L9_50M.json --labels layer3_50M layer6_50M layer9_50M --output dashboards/layer_comparison_50M.json
