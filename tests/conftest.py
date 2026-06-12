"""Deterministic seed-42 fixtures; fully offline."""
import pytest

from vae_factors.data.synthetic import generate_panel
from vae_factors.model.train import train_vae


@pytest.fixture(scope="session")
def panel():
    return generate_panel(n_assets=40, n_days=600, n_sectors=4, k_true=3, seed=42)


@pytest.fixture(scope="session")
def trained(panel):
    return train_vae(panel, latent_dim=3, beta=1.0, epochs=120, seed=42)
