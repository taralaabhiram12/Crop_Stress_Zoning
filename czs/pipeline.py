"""Shared analysis pipeline used by every page."""
import numpy as np

from . import BAND_NAMES
from .clustering import build_features, kmeans_zones
from .indices import compute_indices

DEFAULT_FEATURES = ["NDVI", "GNDVI", "NDRE", "SAVI"]


def bands_from_cube(cube):
    return {n: cube[..., i].astype(np.float32) for i, n in enumerate(BAND_NAMES)}


def cube_from_bands(b):
    return np.stack([b[n] for n in BAND_NAMES], -1).astype(np.float32)


def zone_stats(cls, indices):
    means = {n: [float(indices[n][cls == i].mean()) if (cls == i).any() else None for i in range(3)]
             for n in ("NDVI", "NDRE", "NDWI")}
    return {"zone_pct": [float((cls == i).mean() * 100) for i in range(3)], "zone_means": means,
            "mean_ndvi": float(indices["NDVI"].mean()), "px": list(cls.shape)}


def run_analysis(bands, features=DEFAULT_FEATURES, band_feats=(), k=3, smooth=3, seed=0, n_init=10):
    indices = compute_indices(bands)
    X, names, shape = build_features(indices, bands, list(features), list(band_feats))
    cls, cid, _, _ = kmeans_zones(X, shape, k=k, ref_col=0, seed=seed, smooth=smooth)
    return dict(cls=cls, indices=indices, stats=zone_stats(cls, indices),
                params=dict(features=list(features), bands=list(band_feats), k=k, smooth=smooth, seed=seed))
