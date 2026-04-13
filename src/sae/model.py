"""Top-k Sparse Autoencoder.

Implements the architecture from Gao et al. 2024, "Scaling and evaluating
sparse autoencoders" (https://arxiv.org/abs/2406.04093):

    pre = W_enc (x - b_dec) + b_enc
    z   = TopK(pre, k)                  # zero all but the k largest pre-activations
    x_hat = W_dec z + b_dec             # W_dec columns constrained to unit norm

The top-k activation gives exact control over L0 sparsity without the
shrinkage bias of L1 penalties. Dead latents (ones that never appear in the
top-k for a long stretch) are revived by the AuxK auxiliary loss, which asks
the top-k_aux *dead* latents to reconstruct the residual error left over by
the main reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TopKSAEConfig:
    d_model: int
    d_sae: int
    k: int = 32
    k_aux: int = 512
    dead_steps_threshold: int = 1000
    aux_loss_coef: float = 1 / 32


class TopKSAE(nn.Module):
    def __init__(self, cfg: TopKSAEConfig):
        super().__init__()
        self.cfg = cfg

        self.W_enc = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))

        # Kaiming init for encoder; decoder is tied-transpose at init, then
        # detached and normalized.
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

        # How many steps since each latent last fired. Buffer so it moves
        # with the module and is checkpointed.
        self.register_buffer(
            "steps_since_fired", torch.zeros(cfg.d_sae, dtype=torch.long)
        )

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def topk(self, pre: torch.Tensor, k: int) -> torch.Tensor:
        vals, idx = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, F.relu(vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return reconstruction, total loss, and diagnostics.

        Shapes: `x` is (B, d_model). Works for any leading batch dim if you
        flatten first; training loop passes 2-D activations.
        """
        pre = self.encode_pre(x)
        z = self.topk(pre, self.cfg.k)
        x_hat = self.decode(z)

        recon_err = x - x_hat
        recon_loss = recon_err.pow(2).mean()

        # AuxK: use the top-k_aux *dead* latents to reconstruct the residual
        # error. This is what keeps dead latents from accumulating.
        with torch.no_grad():
            fired = (z != 0).any(dim=0)
            self.steps_since_fired += 1
            self.steps_since_fired[fired] = 0
            dead_mask = self.steps_since_fired > self.cfg.dead_steps_threshold

        aux_loss = x.new_zeros(())
        num_dead = int(dead_mask.sum())
        if num_dead > 0:
            pre_dead = pre.masked_fill(~dead_mask, float("-inf"))
            k_aux = min(self.cfg.k_aux, num_dead)
            z_aux = self.topk(pre_dead, k_aux)
            err_hat = z_aux @ self.W_dec
            aux_loss = (recon_err.detach() - err_hat).pow(2).mean()

        loss = recon_loss + self.cfg.aux_loss_coef * aux_loss

        # Fraction of variance unexplained — standard SAE quality metric.
        fvu = recon_err.pow(2).sum() / (x - x.mean(0)).pow(2).sum().clamp_min(1e-8)

        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "aux_loss": aux_loss.detach() if torch.is_tensor(aux_loss) else aux_loss,
            "fvu": fvu.detach(),
            "num_dead": torch.tensor(num_dead, device=x.device),
            "x_hat": x_hat,
            "z": z,
        }

    @torch.no_grad()
    def post_step(self) -> None:
        """Call after optimizer.step() to re-unit-norm decoder columns."""
        self._normalize_decoder()
