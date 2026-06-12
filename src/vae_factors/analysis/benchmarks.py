"""
Covariance estimators and the out-of-sample minimum-variance test.

The economic question: does the VAE's nonlinear latent structure produce a better
return covariance than linear baselines? We build Σ̂ four ways on the training window,
form the global minimum-variance portfolio for each, and score realised out-of-sample
volatility and Sharpe. A lower OOS volatility means the covariance generalises better.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.synthetic import ReturnsPanel
from ..model.train import TrainResult


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    x = returns - returns.mean(0)
    return (x.T @ x) / len(returns)


def ledoit_wolf_covariance(returns: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(returns).covariance_


def pca_covariance(returns: np.ndarray, k: int) -> np.ndarray:
    """Rank-k factor covariance: top-k eigenspace + diagonal idiosyncratic residual."""
    x = returns - returns.mean(0)
    S = (x.T @ x) / len(returns)
    w, V = np.linalg.eigh(S)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    Vk, wk = V[:, :k], np.clip(w[:k], 0, None)
    low = Vk @ np.diag(wk) @ Vk.T
    resid = np.clip(np.diag(S) - np.diag(low), 1e-8, None)
    return low + np.diag(resid)


def vae_covariance(res: TrainResult, train_returns: np.ndarray, n_samples: int = 4000,
                   regime_sample: np.ndarray | None = None) -> np.ndarray:
    """Factor-model covariance implied by the VAE: systematic + idiosyncratic.

    * **Systematic** part — Monte-Carlo covariance of the generative model: sample
      z ~ N(0, I), decode to standardised returns, de-standardise, take the sample
      covariance. This is the (nonlinear) factor structure the decoder has learned.
    * **Idiosyncratic** part — the per-stock variance of the reconstruction residual
      on the training data, added to the diagonal. Without it the diagonal is
      under-estimated and the min-variance optimiser over-levers into noise.
    """
    model = res.model
    model.eval()
    c_dim = model.cond_dim
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim)
        c = None
        if c_dim:
            base = regime_sample if regime_sample is not None else [1.0, 0.0]
            c = torch.tensor(np.tile(base, (n_samples, 1)), dtype=torch.float32)
        x_std = model.decode(z, c).numpy()
    systematic = sample_covariance(res.scaler.inverse(x_std))

    # idiosyncratic residual variance from training reconstructions
    Xtr = torch.tensor(res.scaler.transform(train_returns).astype(np.float32))
    with torch.no_grad():
        mu, _ = model.encode(Xtr, None if not c_dim else
                             torch.tensor(np.tile([1.0, 0.0], (len(train_returns), 1)),
                                          dtype=torch.float32))
        rec_std = model.decode(mu, None if not c_dim else
                               torch.tensor(np.tile([1.0, 0.0], (len(train_returns), 1)),
                                            dtype=torch.float32)).numpy()
    resid = train_returns - res.scaler.inverse(rec_std)
    idio = resid.var(axis=0)
    return systematic + np.diag(idio)


def min_variance_weights(sigma: np.ndarray) -> np.ndarray:
    n = sigma.shape[0]
    inv = np.linalg.inv(sigma + 1e-8 * np.eye(n))
    ones = np.ones(n)
    w = inv @ ones
    return w / (ones @ w)


def evaluate_min_variance(panel: ReturnsPanel, res: TrainResult,
                          train_frac: float = 0.7) -> dict:
    """Build each Σ̂ on the train window, score OOS min-variance vol & Sharpe."""
    split = int(panel.n_days * train_frac)
    Rtr, Rte = panel.returns[:split], panel.returns[split:]
    k = res.model.latent_dim

    estimators = {
        "Sample": sample_covariance(Rtr),
        "Ledoit-Wolf": ledoit_wolf_covariance(Rtr),
        f"PCA (k={k})": pca_covariance(Rtr, k),
        "VAE": vae_covariance(res, Rtr),
    }
    out = {}
    for name, sigma in estimators.items():
        w = min_variance_weights(sigma)
        port = Rte @ w
        vol = float(port.std() * np.sqrt(252))
        sharpe = float(port.mean() / (port.std() + 1e-12) * np.sqrt(252))
        out[name] = {"oos_vol": vol, "oos_sharpe": sharpe,
                     "gross_leverage": float(np.abs(w).sum())}
    return out
