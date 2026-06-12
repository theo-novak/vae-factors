"""Interpret the latent dimensions against known variables and the true factors."""
from __future__ import annotations

import numpy as np

from ..data.synthetic import ReturnsPanel
from ..model.train import TrainResult


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = (a.std() * b.std())
    return float((a * b).mean() / d) if d > 1e-12 else 0.0


def latent_observable_correlations(res: TrainResult, panel: ReturnsPanel) -> dict:
    """|correlation| of each latent dimension with each observable macro variable."""
    Z = res.latent_mu
    obs = {"market": panel.market, "vol_index": panel.vol_index, "slope": panel.slope}
    table = {}
    for j in range(Z.shape[1]):
        table[f"z{j}"] = {name: abs(_corr(Z[:, j], v)) for name, v in obs.items()}
    return table


def factor_recovery(res: TrainResult, panel: ReturnsPanel) -> dict:
    """How well the latent space spans the true factor space.

    Regress each true factor on the latent means; the R² is the share of that factor's
    variation the VAE has captured (akin to a canonical-correlation alignment).
    """
    Z = res.latent_mu
    Zc = np.column_stack([np.ones(len(Z)), Z])
    out = {}
    for j in range(panel.factors.shape[1]):
        y = panel.factors[:, j]
        beta, *_ = np.linalg.lstsq(Zc, y, rcond=None)
        pred = Zc @ beta
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        out[f"true_factor_{j}"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return out


def stock_latent_betas(res: TrainResult, panel: ReturnsPanel) -> np.ndarray:
    """Per-stock sensitivity to each latent factor (regression of returns on Z).

    Returns (N, k) — used as the feature for the stock embedding.
    """
    Z = res.latent_mu
    Zc = np.column_stack([np.ones(len(Z)), Z])
    R = res.scaler.transform(panel.returns)
    betas, *_ = np.linalg.lstsq(Zc, R, rcond=None)   # (k+1, N)
    return betas[1:].T                                # (N, k)
