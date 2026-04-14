"""Per-feature max-activating-example extraction.

Given a trained TopKSAE and the same base LM it was trained on, stream text
through the model, compute SAE pre-activations at each token, and keep a
running top-N heap per latent of (activation, doc_id, token_pos, context).
The output is a JSON file consumable by a notebook or static viewer.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae.model import TopKSAE, TopKSAEConfig


@dataclass
class Example:
    activation: float
    doc_id: int
    token_pos: int
    tokens: list[int]
    highlight: int

    def to_dict(self, tokenizer) -> dict:
        return {
            "activation": self.activation,
            "doc_id": self.doc_id,
            "token_pos": self.token_pos,
            "text": tokenizer.decode(self.tokens),
            "highlight_token": tokenizer.decode([self.tokens[self.highlight]]),
        }


def load_sae(checkpoint_path: str, device: str = "cuda") -> TopKSAE:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = TopKSAEConfig(**ckpt["cfg"])
    sae = TopKSAE(cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def _get_blocks(model):
    for path in (("gpt_neox", "layers"), ("model", "layers"), ("transformer", "h")):
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok:
            return cur
    raise RuntimeError("could not locate transformer block list on model")


@torch.no_grad()
def collect_max_activating(
    sae: TopKSAE,
    model_name: str,
    layer: int,
    output_path: str,
    top_n: int = 16,
    context_radius: int = 16,
    num_docs: int = 2000,
    context_len: int = 512,
    dataset_name: str = "monology/pile-uncopyrighted",
    device: str = "cuda",
    dataset=None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16
    ).to(device)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = _get_blocks(model)[layer].register_forward_hook(hook)

    d_sae = sae.cfg.d_sae
    # Min-heaps of (activation, counter, Example) per latent. Counter breaks
    # ties so heapq never compares Example objects.
    heaps: list[list] = [[] for _ in range(d_sae)]
    tie = 0

    if dataset is None:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    for doc_id, doc in enumerate(dataset):
        if doc_id >= num_docs:
            break
        text = doc.get("text") or ""
        if not text:
            continue
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=context_len
        ).to(device)
        model(**enc)
        hidden = captured["h"][0].float()  # (T, d_model)
        pre = sae.encode_pre(hidden)  # (T, d_sae)
        z = sae.topk(pre, sae.cfg.k)  # only keep values that survive top-k

        # For each token position, find which latents fired and update heaps.
        nz_pos, nz_lat = z.nonzero(as_tuple=True)
        vals = z[nz_pos, nz_lat]
        token_ids = enc["input_ids"][0].tolist()

        for pos, lat, val in zip(
            nz_pos.tolist(), nz_lat.tolist(), vals.tolist(), strict=True
        ):
            lo = max(0, pos - context_radius)
            hi = min(len(token_ids), pos + context_radius + 1)
            ex = Example(
                activation=val,
                doc_id=doc_id,
                token_pos=pos,
                tokens=token_ids[lo:hi],
                highlight=pos - lo,
            )
            h = heaps[lat]
            tie += 1
            if len(h) < top_n:
                heapq.heappush(h, (val, tie, ex))
            elif val > h[0][0]:
                heapq.heapreplace(h, (val, tie, ex))

    handle.remove()

    out: dict[str, list[dict]] = {}
    for lat, h in enumerate(heaps):
        if not h:
            continue
        examples = sorted(h, key=lambda t: -t[0])
        out[str(lat)] = [ex.to_dict(tokenizer) for _, _, ex in examples]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
