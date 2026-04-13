import torch

from sae.model import TopKSAE, TopKSAEConfig


def test_topk_sparsity():
    cfg = TopKSAEConfig(d_model=32, d_sae=128, k=4)
    sae = TopKSAE(cfg)
    x = torch.randn(16, 32)
    out = sae(x)
    z = out["z"]
    assert z.shape == (16, 128)
    # Exactly k nonzeros per row (ReLU of top-k may zero some, so <= k).
    assert (z != 0).sum(dim=-1).max().item() <= 4


def test_decoder_unit_norm_after_step():
    cfg = TopKSAEConfig(d_model=16, d_sae=64, k=4)
    sae = TopKSAE(cfg)
    opt = torch.optim.SGD(sae.parameters(), lr=0.1)
    x = torch.randn(8, 16)
    out = sae(x)
    out["loss"].backward()
    opt.step()
    sae.post_step()
    norms = sae.W_dec.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
