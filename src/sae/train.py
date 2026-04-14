"""SAE training loop."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from sae.activations import ActivationStream
from sae.model import TopKSAE, TopKSAEConfig


def train(
    model_name: str = "EleutherAI/pythia-160m",
    layer: int = 6,
    d_sae: int = 16_384,
    k: int = 32,
    total_tokens: int = 50_000_000,
    lr: float = 3e-4,
    batch_size: int = 4096,
    checkpoint_dir: str = "checkpoints",
    log_every: int = 50,
    device: str = "cuda",
    tag: str = "",
):
    stream = ActivationStream(
        model_name=model_name, layer=layer, device=device, batch_out=batch_size
    )
    cfg = TopKSAEConfig(d_model=stream.d_model, d_sae=d_sae, k=k)
    sae = TopKSAE(cfg).to(device)

    # Adam with no weight decay: we want unit-norm decoder columns to be
    # maintained exactly by the post_step projection, not nudged by L2.
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    ckpt = Path(checkpoint_dir)
    ckpt.mkdir(exist_ok=True, parents=True)

    seen = 0
    step = 0
    pbar = tqdm(total=total_tokens, unit="tok", unit_scale=True)
    for batch in stream:
        out = sae(batch)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae.post_step()

        seen += batch.shape[0]
        step += 1
        pbar.update(batch.shape[0])

        if step % log_every == 0:
            pbar.set_postfix(
                fvu=f"{out['fvu'].item():.3f}",
                recon=f"{out['recon_loss'].item():.3f}",
                dead=int(out["num_dead"].item()),
            )

        if seen >= total_tokens:
            break

    suffix = f"_{tag}" if tag else ""
    torch.save(
        {"state_dict": sae.state_dict(), "cfg": cfg.__dict__},
        ckpt / f"sae_L{layer}_d{d_sae}_k{k}{suffix}.pt",
    )
    stream.close()
    pbar.close()
