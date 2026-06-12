import numpy as np

from vae_factors.data.synthetic import generate_panel


def test_shapes(panel):
    assert panel.returns.shape == (600, 40)
    assert panel.loadings.shape == (40, 3)
    assert panel.factors.shape == (600, 3)
    assert panel.regime.shape == (600,)
    assert set(np.unique(panel.regime)).issubset({0, 1})


def test_reproducible():
    a = generate_panel(n_assets=20, n_days=200, seed=42)
    b = generate_panel(n_assets=20, n_days=200, seed=42)
    assert np.allclose(a.returns, b.returns)


def test_market_factor_drives_returns(panel):
    # the cross-sectional mean return should correlate strongly with the market factor
    mean_ret = panel.returns.mean(axis=1)
    c = np.corrcoef(mean_ret, panel.market)[0, 1]
    assert c > 0.8


def test_some_anomalies(panel):
    assert 0 < panel.anomalous.sum() < panel.n_assets
