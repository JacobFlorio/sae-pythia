"""Streaming residual-stream activation collector.

Hooks a chosen layer of a HF causal LM, runs text through it, and yields
flattened (B, d_model) activation batches. Keeps a rolling buffer so that
activations from many forward passes are shuffled together before being
handed to the SAE — this matters because adjacent tokens in a document are
highly correlated, and a naive in-order stream gives the SAE a degenerate
distribution to fit.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class ActivationStream:
    def __init__(
        self,
        model_name: str,
        layer: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        context_len: int = 512,
        buffer_tokens: int = 262_144,
        batch_out: int = 4096,
        dataset_name: str = "monology/pile-uncopyrighted",
        dataset_split: str = "train",
    ):
        self.device = device
        self.dtype = dtype
        self.context_len = context_len
        self.buffer_tokens = buffer_tokens
        self.batch_out = batch_out

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self.model.eval()

        self.d_model = self.model.config.hidden_size
        self._captured: torch.Tensor | None = None

        # GPTNeoX (Pythia): model.gpt_neox.layers[i]. Generalize by walking
        # the config, but Pythia is the default target so hardcode the path
        # and fall back to `model.model.layers` for Llama-likes.
        blocks = self._get_blocks()
        self._hook = blocks[layer].register_forward_hook(self._hook_fn)

        self.dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)

    def _get_blocks(self):
        m = self.model
        for path in (
            ("gpt_neox", "layers"),
            ("model", "layers"),
            ("transformer", "h"),
        ):
            cur = m
            ok = True
            for attr in path:
                if not hasattr(cur, attr):
                    ok = False
                    break
                cur = getattr(cur, attr)
            if ok:
                return cur
        raise RuntimeError("could not locate transformer block list on model")

    def _hook_fn(self, _module, _inputs, output):
        # HF blocks return either a Tensor or a tuple whose first element is
        # the residual stream. Keep the residual stream only.
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured = hidden.detach()

    def __iter__(self) -> Iterator[torch.Tensor]:
        buffer = torch.empty(
            (self.buffer_tokens, self.d_model), dtype=self.dtype, device=self.device
        )
        filled = 0
        doc_iter = iter(self.dataset)

        while True:
            while filled < self.buffer_tokens:
                try:
                    doc = next(doc_iter)
                except StopIteration:
                    return
                text = doc.get("text") or ""
                if not text:
                    continue
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.context_len,
                ).to(self.device)
                with torch.no_grad():
                    self.model(**tokens)
                acts = self._captured  # (1, T, d_model)
                if acts is None:
                    continue
                flat = acts.reshape(-1, self.d_model)
                take = min(flat.shape[0], self.buffer_tokens - filled)
                buffer[filled : filled + take] = flat[:take]
                filled += take

            perm = torch.randperm(filled, device=self.device)
            buffer[:filled] = buffer[perm]

            # Yield out ~3/4 of the buffer, leaving 1/4 to mix with the next
            # refill. This is the "shuffling buffer" trick from the SAE
            # training literature.
            yield_end = (filled * 3) // 4
            for start in range(0, yield_end, self.batch_out):
                end = min(start + self.batch_out, yield_end)
                yield buffer[start:end].float()

            remaining = filled - yield_end
            buffer[:remaining] = buffer[yield_end:filled]
            filled = remaining

    def close(self) -> None:
        self._hook.remove()
