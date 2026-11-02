# VAE Factors — variational autoencoder factor model for equity covariances

A variational autoencoder learns a low-dimensional latent representation of a cross-section of daily returns. The trained generative model gives a covariance estimate, which is tested where it matters: out-of-sample global minimum-variance portfolios. It is compared with the sample covariance, Ledoit–Wolf shrinkage and a PCA factor model. Options include a β-VAE (KL weight) and a regime-conditional CVAE.

The data is **synthetic and offline**: a seed-42 panel generated from known latent factors (market, sector/style, and one tied to a macro series), so factor recovery can be checked against ground truth. A fraction of names carry episodic shocks followed by negative drift, which tests whether reconstruction error flags later drawdowns. Optional fetchers in `data/fetchers.py` can pull real data.

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
# Defaults: 60 assets, 1000 days, latent k=3, beta=1
vae train

# beta-VAE with a regime-conditional decoder
vae train --latent 5 --beta 0.5 --conditional

# Interactive dashboard
streamlit run src/vae_factors/app.py
```

`train` reports four sets of results:

- out-of-sample minimum-variance volatility, Sharpe ratio and gross leverage for each covariance estimator
- R² recovery of the true factors
- correlations between latent dimensions and the observable market, volatility and slope series
- the anomaly test: mean drawdown of high-error versus low-error names

## Layout

| Path | Contents |
|---|---|
| `data/` | Synthetic panel, fetchers, DuckDB store, schemas |
| `model/` | `vae.py` (VAE/CVAE), `train.py` |
| `analysis/` | `benchmarks.py` (sample, Ledoit–Wolf, PCA, VAE covariance; min-variance test), `interpret.py`, `anomaly.py`, `embedding.py` |
| `report/` | Plotly figures used by the dashboard |

## Tests

```bash
pytest -q
```

## Limits

- Factor recovery and the anomaly test are measured on a panel built to contain those effects. They show that the method works when the structure exists, not that the structure exists in real markets.
- Run the CLI to get the numbers. This README deliberately quotes none.
