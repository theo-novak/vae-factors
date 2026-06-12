"""
Variational autoencoder over the daily cross-section of returns.

Each trading day's vector of ``N`` standardised stock returns is treated as one
observation. The encoder maps it to a Gaussian ``q(z|x) = N(μ, diag σ²)`` in a
``k``-dimensional latent space; the decoder reconstructs the ``N`` returns from a
sampled ``z``. The latent dimensions are therefore **data-driven daily factors**:
``z_t`` is the factor realisation on day ``t`` and the decoder columns are the factor
loadings.

Loss = reconstruction MSE + β · KL(q(z|x) ‖ N(0, I)). Sweeping β trades reconstruction
fidelity against latent disentanglement (β-VAE, Higgins et al. 2017).

``ConditionalVAE`` appends a one-hot macro-regime label to both encoder input and
decoder input, so the learned loadings can shift between regimes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, n_assets: int, latent_dim: int = 3, hidden: int = 64,
                 cond_dim: int = 0):
        super().__init__()
        self.n_assets = n_assets
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.enc = nn.Sequential(
            nn.Linear(n_assets + cond_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_assets),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor | None = None):
        h = self.enc(self._cat(x, c))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self.dec(self._cat(z, c))

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        return self.decode(z, c), mu, logvar, z

    @staticmethod
    def _cat(a: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        return a if c is None else torch.cat([a, c], dim=-1)


def vae_loss(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 1.0):
    """Return (total, reconstruction, KL). Reconstruction is summed-per-sample MSE."""
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon + beta * kl, recon, kl
