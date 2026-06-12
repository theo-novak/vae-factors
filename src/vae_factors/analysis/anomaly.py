"""
Reconstruction error as an anomaly signal.

Stocks the encoder/decoder cannot explain — those with high reconstruction error —
are carrying variation outside the learned factor structure. We test whether high
reconstruction error flags the names that subsequently draw down (the synthetic panel
plants episodic shocks + negative drift in a subset of names).
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.synthetic import ReturnsPanel
from ..model.train import TrainResult


def reconstruction_error(res: TrainResult, panel: ReturnsPanel
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Return (per_stock_error (N,), per_cell_error (T, N)) on standardised returns."""
    model = res.model
    model.eval()
    X = torch.tensor(res.scaler.transform(panel.returns).astype(np.float32))
    c = None
    if model.cond_dim:
        oh = np.zeros((panel.n_days, 2), np.float32)
        oh[np.arange(panel.n_days), panel.regime] = 1.0
        c = torch.tensor(oh)
    with torch.no_grad():
        mu, _ = model.encode(X, c)
        x_hat = model.decode(mu, c)
    cell = ((x_hat - X) ** 2).numpy()
    return cell.mean(0), cell


def _idiosyncratic_drawdown(panel: ReturnsPanel) -> np.ndarray:
    """Worst peak-to-trough drawdown of each stock's *market-residual* return path.

    Stripping the market factor isolates idiosyncratic risk — the channel the planted
    anomalies live in — from systematic moves shared by every name.
    """
    mkt = panel.market - panel.market.mean()
    var = float((mkt ** 2).mean()) + 1e-12
    resid = np.empty_like(panel.returns)
    for i in range(panel.n_assets):
        beta = float((panel.returns[:, i] * mkt).mean()) / var
        resid[:, i] = panel.returns[:, i] - beta * panel.market
    cum = np.cumprod(1 + resid, axis=0)
    roll_max = np.maximum.accumulate(cum, axis=0)
    return ((cum - roll_max) / roll_max).min(axis=0)


def anomaly_drawdown_test(res: TrainResult, panel: ReturnsPanel,
                          top_frac: float = 0.2) -> dict:
    """Do high-reconstruction-error names suffer worse *idiosyncratic* drawdowns?

    Splits the universe at the ``top_frac`` error quantile and compares the worst
    market-neutral drawdown of the two groups, plus the rank correlation of error with
    the planted-anomaly flag.
    """
    err, _ = reconstruction_error(res, panel)
    idio_dd = _idiosyncratic_drawdown(panel)

    thresh = np.quantile(err, 1 - top_frac)
    high = err >= thresh
    low = err < thresh

    from scipy.stats import spearmanr
    rho, _ = spearmanr(err, panel.anomalous.astype(float))
    return {
        "high_error_mean_dd": float(idio_dd[high].mean()),
        "low_error_mean_dd": float(idio_dd[low].mean()),
        "error_vs_anomaly_rho": float(rho),
        "n_high": int(high.sum()),
        "per_stock_error": err,
        "max_drawdown": idio_dd,
    }
