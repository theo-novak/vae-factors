"""Training driver and β-sweep for the returns VAE."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.synthetic import ReturnsPanel
from .vae import VAE, vae_loss


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class TrainResult:
    model: VAE
    scaler: Scaler
    latent_mu: np.ndarray             # (T, k) posterior means over the whole sample
    history: dict = field(default_factory=dict)
    beta: float = 1.0


def _one_hot(regime: np.ndarray) -> np.ndarray:
    oh = np.zeros((len(regime), 2), dtype=np.float32)
    oh[np.arange(len(regime)), regime] = 1.0
    return oh


def train_vae(panel: ReturnsPanel, latent_dim: int = 3, beta: float = 1.0,
              hidden: int = 64, epochs: int = 300, lr: float = 2e-3,
              conditional: bool = False, train_frac: float = 0.7,
              seed: int = 42) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    R = panel.returns.astype(np.float32)
    split = int(panel.n_days * train_frac)
    mean, std = R[:split].mean(0), R[:split].std(0) + 1e-8
    scaler = Scaler(mean=mean, std=std)
    Xall = scaler.transform(R)
    Xtr = torch.tensor(Xall[:split])

    cond_dim = 2 if conditional else 0
    Call = _one_hot(panel.regime) if conditional else None
    Ctr = torch.tensor(Call[:split]) if conditional else None

    model = VAE(panel.n_assets, latent_dim=latent_dim, hidden=hidden, cond_dim=cond_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"total": [], "recon": [], "kl": []}
    for _ in range(epochs):
        opt.zero_grad()
        x_hat, mu, logvar, _ = model(Xtr, Ctr)
        total, recon, kl = vae_loss(x_hat, Xtr, mu, logvar, beta)
        total.backward()
        opt.step()
        hist["total"].append(total.item()); hist["recon"].append(recon.item())
        hist["kl"].append(kl.item())

    model.eval()
    with torch.no_grad():
        mu_all, _ = model.encode(torch.tensor(Xall),
                                 torch.tensor(Call) if conditional else None)
    return TrainResult(model=model, scaler=scaler, latent_mu=mu_all.numpy(),
                       history=hist, beta=beta)


def beta_sweep(panel: ReturnsPanel, betas=(0.1, 0.5, 1.0, 2.0, 4.0, 8.0),
               latent_dim: int = 3, epochs: int = 200, seed: int = 42) -> dict:
    """Reconstruction-vs-KL trade-off across β (the disentanglement-fidelity curve)."""
    rows = []
    for b in betas:
        res = train_vae(panel, latent_dim=latent_dim, beta=b, epochs=epochs, seed=seed)
        rows.append({"beta": b, "recon": res.history["recon"][-1],
                     "kl": res.history["kl"][-1]})
    return {"betas": list(betas), "rows": rows}
