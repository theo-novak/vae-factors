"""2-D embedding of stocks in the learned latent loading space."""
from __future__ import annotations

import numpy as np

from ..data.synthetic import ReturnsPanel
from ..model.train import TrainResult
from .interpret import stock_latent_betas


def embed_stocks(res: TrainResult, panel: ReturnsPanel, method: str = "auto",
                 seed: int = 42) -> np.ndarray:
    """Project per-stock latent betas to 2-D. Uses UMAP when available, else PCA."""
    feats = stock_latent_betas(res, panel)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)

    if method in ("auto", "umap"):
        try:
            import umap
            n = len(feats)
            reducer = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.1,
                                random_state=seed)
            return reducer.fit_transform(feats)
        except Exception:
            if method == "umap":
                raise
    # PCA fallback
    x = feats - feats.mean(0)
    _, _, Vt = np.linalg.svd(x, full_matrices=False)
    return x @ Vt[:2].T
