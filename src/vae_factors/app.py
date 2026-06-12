"""Server-side Streamlit dashboard (full torch + sklearn + umap stack)."""
from __future__ import annotations

import streamlit as st

from vae_factors.analysis.anomaly import anomaly_drawdown_test, reconstruction_error
from vae_factors.analysis.benchmarks import evaluate_min_variance
from vae_factors.analysis.embedding import embed_stocks
from vae_factors.analysis.interpret import latent_observable_correlations
from vae_factors.data.synthetic import generate_panel
from vae_factors.model.train import beta_sweep, train_vae
from vae_factors.report import plots

st.set_page_config(layout="wide", page_title="VAE Factor Model")
st.title("Variational Autoencoder Factor Model")

with st.sidebar:
    n_assets = st.slider("Assets", 30, 100, 60, 10)
    n_days = st.slider("Days", 500, 1500, 1000, 100)
    latent = st.slider("Latent dim k", 2, 6, 3, 1)
    beta = st.select_slider("β (KL weight)", [0.1, 0.5, 1.0, 2.0, 4.0, 8.0], value=1.0)
    epochs = st.slider("Epochs", 100, 500, 300, 50)
    conditional = st.checkbox("Conditional VAE (macro regime)", value=False)
    run_sweep = st.checkbox("Run β-sweep (slower)", value=False)
    run = st.button("Train VAE", type="primary")

if run:
    with st.spinner("Training VAE…"):
        panel = generate_panel(n_assets=n_assets, n_days=n_days)
        res = train_vae(panel, latent_dim=latent, beta=beta, epochs=epochs,
                        conditional=conditional)
        scores = evaluate_min_variance(panel, res)
        corr = latent_observable_correlations(res, panel)
        coords = embed_stocks(res, panel)
        anom = anomaly_drawdown_test(res, panel)
        _, cell_err = reconstruction_error(res, panel)
        sweep = beta_sweep(panel, latent_dim=latent, epochs=min(epochs, 200)) if run_sweep else None

    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.loss_history(res.history), width="stretch")
    c2.plotly_chart(plots.latent_trajectories(res.latent_mu), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(plots.correlation_heatmap(corr), width="stretch")
    c4.plotly_chart(plots.min_variance_bars(scores), width="stretch")
    st.plotly_chart(plots.embedding_scatter(coords, panel.sectors), width="stretch")
    st.plotly_chart(plots.recon_error_heatmap(cell_err, panel.anomalous), width="stretch")
    st.metric("Anomaly: high-error vs low-error mean drawdown",
              f"{anom['high_error_mean_dd']:.1%} vs {anom['low_error_mean_dd']:.1%}")
    if sweep is not None:
        st.plotly_chart(plots.beta_tradeoff(sweep), width="stretch")
else:
    st.info("Configure in the sidebar and click **Train VAE**.")
