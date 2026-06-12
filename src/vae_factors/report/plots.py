"""Plotly figures for the VAE factor model."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]


def loss_history(history: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["recon"], name="Reconstruction",
                             line=dict(color=PALETTE[0])))
    fig.add_trace(go.Scatter(y=history["kl"], name="KL", line=dict(color=PALETTE[3]),
                             yaxis="y2"))
    fig.update_layout(template="plotly_dark", height=380, title="VAE training",
                      xaxis_title="Epoch",
                      yaxis=dict(title="Reconstruction MSE", color=PALETTE[0]),
                      yaxis2=dict(title="KL", overlaying="y", side="right", color=PALETTE[3]),
                      legend=dict(orientation="h", y=1.12))
    return fig


def beta_tradeoff(sweep: dict) -> go.Figure:
    recon = [r["recon"] for r in sweep["rows"]]
    kl = [r["kl"] for r in sweep["rows"]]
    fig = go.Figure(go.Scatter(x=kl, y=recon, mode="markers+lines+text",
        text=[f"β={b}" for b in sweep["betas"]], textposition="top center",
        marker=dict(size=10, color=PALETTE[2])))
    fig.update_layout(template="plotly_dark", height=380,
        title="Disentanglement–fidelity trade-off", xaxis_title="KL (disentanglement →)",
        yaxis_title="Reconstruction MSE (fidelity ↓ better)")
    return fig


def latent_trajectories(latent_mu: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for j in range(latent_mu.shape[1]):
        fig.add_trace(go.Scatter(y=latent_mu[:, j], name=f"z{j}",
                                 line=dict(color=PALETTE[j % len(PALETTE)])))
    fig.update_layout(template="plotly_dark", height=380, title="Latent factor trajectories",
                      xaxis_title="Day", yaxis_title="Posterior mean",
                      legend=dict(orientation="h", y=1.12))
    return fig


def correlation_heatmap(table: dict) -> go.Figure:
    dims = list(table.keys())
    obs = list(next(iter(table.values())).keys())
    Z = np.array([[table[d][o] for o in obs] for d in dims])
    fig = go.Figure(go.Heatmap(z=Z, x=obs, y=dims, colorscale="Viridis",
                               zmin=0, zmax=1, text=np.round(Z, 2),
                               texttemplate="%{text}"))
    fig.update_layout(template="plotly_dark", height=340,
                      title="|corr| latent dim vs observable")
    return fig


def min_variance_bars(scores: dict) -> go.Figure:
    names = list(scores.keys())
    vols = [scores[n]["oos_vol"] for n in names]
    fig = go.Figure(go.Bar(x=names, y=vols, marker_color=PALETTE[:len(names)],
        text=[f"{v:.1%}" for v in vols], textposition="outside"))
    fig.update_layout(template="plotly_dark", height=380,
        title="Out-of-sample minimum-variance portfolio volatility (lower = better)",
        yaxis_title="Annualised vol", yaxis_tickformat=".0%")
    return fig


def recon_error_heatmap(cell_error: np.ndarray, anomalous: np.ndarray | None = None,
                        n_time_bins: int = 60) -> go.Figure:
    """Reconstruction error over (time × stock).

    ``cell_error`` is the per-cell squared error ``(T, N)`` from
    ``anomaly.reconstruction_error``. Stocks are ordered by total error (worst on
    the left) and time is averaged into ``n_time_bins`` columns so the planted
    shock episodes show up as bright vertical streaks.
    """
    T, N = cell_error.shape
    order = np.argsort(-cell_error.mean(0))            # worst-reconstructed first
    err = cell_error[:, order]
    bins = min(n_time_bins, T)
    edges = np.linspace(0, T, bins + 1).astype(int)
    binned = np.array([err[edges[i]:edges[i + 1]].mean(0) for i in range(bins)])  # (bins, N)

    ylabels = [f"S{order[i]}" for i in range(N)]
    if anomalous is not None:
        ylabels = [f"{'★ ' if anomalous[order[i]] else ''}S{order[i]}" for i in range(N)]

    fig = go.Figure(go.Heatmap(z=binned.T, x=np.linspace(0, T, bins), y=ylabels,
                               colorscale="Inferno", colorbar=dict(title="MSE")))
    fig.update_layout(template="plotly_dark", height=460,
                      title="Reconstruction error heatmap (stocks sorted worst→best; "
                            "★ = planted anomaly)",
                      xaxis_title="Day", yaxis_title="Stock")
    return fig


def embedding_scatter(coords: np.ndarray, sectors: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for s in np.unique(sectors):
        m = sectors == s
        fig.add_trace(go.Scatter(x=coords[m, 0], y=coords[m, 1], mode="markers",
            name=f"Sector {s}", marker=dict(size=9, color=PALETTE[s % len(PALETTE)])))
    fig.update_layout(template="plotly_dark", height=420,
        title="Stocks embedded in latent loading space (coloured by sector)",
        xaxis_title="dim 1", yaxis_title="dim 2")
    return fig
