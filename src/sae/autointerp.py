"""Automated interpretability: feature -> natural-language description -> score.

Two-stage pipeline in the spirit of Bills et al. 2023:

1. **Explain.** Show Claude the top-activating examples for a latent, with the
   peak-activation token highlighted. Ask for a short description of what the
   feature appears to detect.

2. **Score.** Give Claude the description plus a held-out mix of
   high-activating and random ("distractor") snippets, and ask which ones it
   predicts will activate. Compare predictions against ground truth; report
   per-feature balanced accuracy.

The scorer's balanced accuracy on this forced-choice task is the headline
auto-interp metric — random guessing is 0.5, perfect prediction is 1.0.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass

import anthropic


EXPLAINER_SYSTEM = (
    "You are an interpretability researcher. You will be shown short text "
    "snippets where one token is marked with <<token>>. These are snippets "
    "where a particular neuron in a language model fired strongly on the "
    "marked token. Your job is to describe, in one sentence, what the neuron "
    "appears to detect. Be specific. Respond with only the description."
)

SCORER_SYSTEM = (
    "You are an interpretability researcher evaluating a hypothesis about "
    "what a neuron detects. You will be given a description of the neuron "
    "and a numbered list of snippets, each with one token marked <<token>>. "
    "For each snippet, predict whether the neuron fires strongly on the "
    "marked token. Respond with a JSON array of booleans, one per snippet, "
    "in order. No prose."
)


@dataclass
class AutoInterpResult:
    latent: int
    description: str
    balanced_accuracy: float
    n_positive: int
    n_negative: int


def _format_snippet(example: dict) -> str:
    text = example["text"]
    hl = example["highlight_token"]
    # Mark the first occurrence of the highlight token in the snippet.
    idx = text.find(hl)
    if idx < 0:
        return text
    return text[:idx] + f"<<{hl}>>" + text[idx + len(hl) :]


def explain_feature(
    client: anthropic.Anthropic,
    examples: list[dict],
    model: str = "claude-sonnet-4-5",
    max_examples: int = 12,
) -> str:
    snippets = "\n\n".join(
        f"{i + 1}. {_format_snippet(ex)}"
        for i, ex in enumerate(examples[:max_examples])
    )
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        system=EXPLAINER_SYSTEM,
        messages=[{"role": "user", "content": snippets}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()


def score_feature(
    client: anthropic.Anthropic,
    description: str,
    positives: list[dict],
    negatives: list[dict],
    model: str = "claude-sonnet-4-5",
    seed: int = 0,
) -> float:
    """Balanced accuracy on a forced-choice positive/negative discrimination."""
    rng = random.Random(seed)
    labeled = [(ex, True) for ex in positives] + [(ex, False) for ex in negatives]
    rng.shuffle(labeled)

    snippets = "\n\n".join(
        f"{i + 1}. {_format_snippet(ex)}" for i, (ex, _) in enumerate(labeled)
    )
    user = f"Description: {description}\n\nSnippets:\n{snippets}"

    msg = client.messages.create(
        model=model,
        max_tokens=400,
        system=SCORER_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    match = re.search(r"\[[^\]]*\]", raw)
    if not match:
        return float("nan")
    try:
        preds = json.loads(match.group(0))
    except json.JSONDecodeError:
        return float("nan")
    if len(preds) != len(labeled):
        return float("nan")

    tp = sum(1 for (_, y), p in zip(labeled, preds, strict=True) if y and p)
    tn = sum(1 for (_, y), p in zip(labeled, preds, strict=True) if not y and not p)
    pos = sum(1 for _, y in labeled if y)
    neg = len(labeled) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return 0.5 * (tp / pos + tn / neg)


def run_autointerp(
    dashboard_path: str,
    output_path: str,
    num_features: int | None = None,
    n_score_positive: int = 4,
    n_score_negative: int = 4,
    model: str = "claude-sonnet-4-5",
    seed: int = 0,
) -> list[AutoInterpResult]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    with open(dashboard_path) as f:
        dashboard: dict[str, list[dict]] = json.load(f)

    client = anthropic.Anthropic()
    latents = list(dashboard.keys())
    if num_features is not None:
        latents = latents[:num_features]

    # Build the distractor pool from OTHER features' top examples — these are
    # snippets we know are high-activating for *some* feature, which makes the
    # discrimination task harder (and more meaningful) than random web text.
    rng = random.Random(seed)
    results: list[AutoInterpResult] = []

    for latent in latents:
        examples = dashboard[latent]
        if len(examples) < n_score_positive * 2:
            continue
        explain_set = examples[: max(8, n_score_positive)]
        positives = examples[-n_score_positive:]

        distractor_pool: list[dict] = []
        for other, other_exs in dashboard.items():
            if other == latent:
                continue
            distractor_pool.extend(other_exs[:2])
        if len(distractor_pool) < n_score_negative:
            continue
        negatives = rng.sample(distractor_pool, n_score_negative)

        desc = explain_feature(client, explain_set, model=model)
        acc = score_feature(client, desc, positives, negatives, model=model, seed=seed)

        results.append(
            AutoInterpResult(
                latent=int(latent),
                description=desc,
                balanced_accuracy=acc,
                n_positive=n_score_positive,
                n_negative=n_score_negative,
            )
        )

    with open(output_path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    return results
