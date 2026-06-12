"""Typer CLI for the VAE factor model."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .analysis.anomaly import anomaly_drawdown_test
from .analysis.benchmarks import evaluate_min_variance
from .analysis.interpret import factor_recovery, latent_observable_correlations
from .data.synthetic import generate_panel
from .model.train import train_vae

app = typer.Typer(add_completion=False, help="Variational autoencoder factor model.")
console = Console()


@app.command()
def train(
    n_assets: int = typer.Option(60),
    n_days: int = typer.Option(1000),
    latent: int = typer.Option(3, help="Latent dimension k."),
    beta: float = typer.Option(1.0, help="KL weight (β-VAE)."),
    epochs: int = typer.Option(300),
    conditional: bool = typer.Option(False, help="Condition on macro regime (CVAE)."),
    seed: int = typer.Option(42),
):
    """Train the VAE and report covariance, interpretation and anomaly diagnostics."""
    panel = generate_panel(n_assets=n_assets, n_days=n_days, seed=seed)
    res = train_vae(panel, latent_dim=latent, beta=beta, epochs=epochs,
                    conditional=conditional, seed=seed)

    cov = evaluate_min_variance(panel, res)
    t = Table(title="Out-of-sample minimum-variance portfolios")
    for c in ("Estimator", "OOS vol", "OOS Sharpe", "Gross lev."):
        t.add_column(c)
    for name, s in cov.items():
        t.add_row(name, f"{s['oos_vol']:.2%}", f"{s['oos_sharpe']:.2f}",
                  f"{s['gross_leverage']:.1f}")
    console.print(t)

    rec = factor_recovery(res, panel)
    console.print("[bold]True-factor recovery (R²):[/] " +
                  "  ".join(f"{k}={v:.2f}" for k, v in rec.items()))

    corr = latent_observable_correlations(res, panel)
    console.print("[bold]Latent ↔ observable |corr|:[/]")
    for d, row in corr.items():
        console.print(f"  {d}: " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()))

    an = anomaly_drawdown_test(res, panel)
    console.print(f"[bold]Anomaly test:[/] high-error mean DD={an['high_error_mean_dd']:.1%} "
                  f"vs low-error {an['low_error_mean_dd']:.1%}; "
                  f"error↔anomaly ρ={an['error_vs_anomaly_rho']:.2f}")


@app.command()
def dashboard():
    console.print("Run: [bold]streamlit run src/vae_factors/app.py[/]")


if __name__ == "__main__":
    app()
