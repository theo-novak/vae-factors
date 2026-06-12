import numpy as np
import torch

from vae_factors.model.vae import VAE, vae_loss


def test_vae_forward_shapes():
    m = VAE(n_assets=12, latent_dim=3, hidden=16)
    x = torch.randn(20, 12)
    x_hat, mu, logvar, z = m(x)
    assert x_hat.shape == (20, 12)
    assert mu.shape == (20, 3) and logvar.shape == (20, 3) and z.shape == (20, 3)


def test_conditional_vae():
    m = VAE(n_assets=8, latent_dim=2, hidden=16, cond_dim=2)
    x = torch.randn(5, 8)
    c = torch.tensor([[1.0, 0.0]] * 5)
    x_hat, mu, logvar, z = m(x, c)
    assert x_hat.shape == (5, 8)


def test_loss_components_nonneg():
    m = VAE(n_assets=10, latent_dim=2, hidden=16)
    x = torch.randn(30, 10)
    x_hat, mu, logvar, _ = m(x)
    total, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=1.0)
    assert recon >= 0 and kl >= -1e-5
    assert torch.isfinite(total)


def test_training_reduces_loss(trained):
    h = trained.history["total"]
    assert h[-1] < h[0]
    assert trained.latent_mu.shape[1] == 3
    assert np.isfinite(trained.latent_mu).all()


def test_beta_sweep_monotone_kl(panel):
    from vae_factors.model.train import beta_sweep

    sweep = beta_sweep(panel, betas=(0.5, 4.0), latent_dim=3, epochs=40)
    assert len(sweep["rows"]) == 2
    lo, hi = sweep["rows"]
    # stronger KL penalty (β=4) compresses the latent code → smaller KL than β=0.5
    assert hi["kl"] <= lo["kl"] + 1e-6
    assert all(np.isfinite(r["recon"]) and np.isfinite(r["kl"]) for r in sweep["rows"])
