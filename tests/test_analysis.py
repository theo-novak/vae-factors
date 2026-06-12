import numpy as np

from vae_factors.analysis.anomaly import anomaly_drawdown_test, reconstruction_error
from vae_factors.analysis.benchmarks import (
    evaluate_min_variance,
    ledoit_wolf_covariance,
    min_variance_weights,
    pca_covariance,
    sample_covariance,
    vae_covariance,
)
from vae_factors.analysis.embedding import embed_stocks
from vae_factors.analysis.interpret import (
    factor_recovery,
    latent_observable_correlations,
    stock_latent_betas,
)


def _psd(s, tol=1e-6):
    return (np.linalg.eigvalsh((s + s.T) / 2) > -tol).all()


def test_covariance_estimators_psd(panel):
    R = panel.returns[:400]
    assert _psd(sample_covariance(R))
    assert _psd(ledoit_wolf_covariance(R))
    assert _psd(pca_covariance(R, 3))


def test_vae_covariance_shape_psd(trained, panel):
    s = vae_covariance(trained, panel.returns[:400], n_samples=1000)
    assert s.shape == (panel.n_assets, panel.n_assets)
    assert _psd(s)


def test_min_variance_weights_sum_to_one(panel):
    s = sample_covariance(panel.returns[:400])
    w = min_variance_weights(s)
    assert abs(w.sum() - 1.0) < 1e-8


def test_evaluate_min_variance_finite(trained, panel):
    scores = evaluate_min_variance(panel, trained)
    assert {"Sample", "Ledoit-Wolf", "VAE"}.issubset(scores)
    for s in scores.values():
        assert np.isfinite(s["oos_vol"]) and s["oos_vol"] > 0


def test_factor_recovery_market(trained, panel):
    rec = factor_recovery(trained, panel)
    # the market factor (factor 0) should be well recovered by the latent space
    assert rec["true_factor_0"] > 0.5


def test_latent_corr_market(trained, panel):
    corr = latent_observable_correlations(trained, panel)
    best_market = max(row["market"] for row in corr.values())
    assert best_market > 0.6


def test_stock_betas_shape(trained, panel):
    b = stock_latent_betas(trained, panel)
    assert b.shape == (panel.n_assets, 3)


def test_anomaly_error_tracks_planted_flag(trained, panel):
    an = anomaly_drawdown_test(trained, panel)
    assert an["error_vs_anomaly_rho"] > 0.0
    # planted anomalies draw down more → high-error group has worse mean drawdown
    assert an["high_error_mean_dd"] <= an["low_error_mean_dd"]


def test_reconstruction_error_shapes(trained, panel):
    per_stock, per_cell = reconstruction_error(trained, panel)
    assert per_stock.shape == (panel.n_assets,)
    assert per_cell.shape == (panel.n_days, panel.n_assets)


def test_embed_stocks_2d(trained, panel):
    coords = embed_stocks(trained, panel, method="auto")
    assert coords.shape == (panel.n_assets, 2)


def test_recon_error_heatmap_dims(trained, panel):
    from vae_factors.report.plots import recon_error_heatmap

    _, cell = reconstruction_error(trained, panel)
    fig = recon_error_heatmap(cell, panel.anomalous, n_time_bins=30)
    z = fig.data[0].z
    # heatmap is (stocks x time_bins); time binned to <= n_time_bins
    assert z.shape[0] == panel.n_assets
    assert z.shape[1] <= 30
