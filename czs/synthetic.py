"""Synthetic multispectral crop field + simulated agronomist field markings."""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

# Typical canopy reflectance (blue, green, red, red-edge, NIR) for healthy vs diseased crop
HEALTHY = np.array([0.04, 0.10, 0.05, 0.30, 0.50])
DISEASED = np.array([0.07, 0.12, 0.18, 0.20, 0.28])


def _smooth_noise(shape, sigma, rng):
    n = gaussian_filter(rng.normal(size=shape), sigma)
    return (n - n.mean()) / (n.std() + 1e-9)


def _severity(size, rng, n_patches):
    yy, xx = np.mgrid[0:size, 0:size]
    s = np.zeros((size, size))
    for _ in range(n_patches):
        cy, cx = rng.uniform(0, size, 2)
        r = rng.uniform(size * 0.08, size * 0.22)
        amp = rng.uniform(0.5, 1.1)
        s += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2)))
    s += 0.12 * _smooth_noise((size, size), size / 10, rng)
    # rank-normalise so every field has a balanced mix of healthy/stressed/diseased
    ranks = s.ravel().argsort().argsort().reshape(s.shape)
    return ranks / (s.size - 1)


def _to_classes(sev, t1=0.38, t2=0.68):
    return np.digitize(sev, [t1, t2]).astype(np.uint8)  # 0 healthy,1 stressed,2 diseased


def generate_field(size=128, seed=42, n_patches=5, noise=0.015, boundary_uncertainty=2.0):
    """Return dict(bands, severity, truth, agronomist) for a synthetic field.

    agronomist = hand-marked zones: same underlying severity but with
    smoothed/jittered boundaries, emulating human field marking.
    """
    rng = np.random.default_rng(seed)
    sev = _severity(size, rng, n_patches)

    mix = sev[..., None]
    refl = HEALTHY * (1 - mix) + DISEASED * mix
    rows = 1 + 0.05 * np.sin(np.arange(size) * 2 * np.pi / 6)[None, :, None]  # crop rows
    illum = 1 + 0.04 * _smooth_noise((size, size), size / 6, rng)[..., None]
    refl = refl * rows * illum
    refl += rng.normal(0, noise, refl.shape) * 0.5
    refl = np.clip(refl, 0.005, 1.0)
    bands = {n: refl[..., i].astype(np.float32) for i, n in
             enumerate(["blue", "green", "red", "rededge", "nir"])}

    jitter = _smooth_noise((size, size), 4, rng) * 0.04 * boundary_uncertainty
    ag = _to_classes(np.clip(sev + jitter, 0, 1))
    ag = median_filter(ag, size=7).astype(np.uint8)
    return {"bands": bands, "severity": sev, "truth": _to_classes(sev), "agronomist": ag}


def rgb_demo(kind="aerial", size=360, seed=1):
    """Synthetic RGB photo for demos/tests: 'aerial' field view or 'leaf' close-up."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    if kind == "leaf":
        img = np.zeros((size, size, 3)) + np.array([236, 236, 230.0])
        a, b, ang = size * 0.44, size * 0.2, np.deg2rad(28)
        cx = cy = size / 2
        u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
        v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
        be = b * (1 - 0.25 * u / a); d = 1 - ((u / a) ** 2 + (v / be) ** 2)
        leaf = d > 0
        base = np.array([62, 138, 48.0]) * (1 - 0.12 * np.abs(v / be))[..., None]
        base[np.abs(v) < 2.5] = [105, 168, 78]
        yw = np.clip((0.2 - d) / 0.2, 0, 1) * (0.5 + 4 * gaussian_filter(rng.normal(size=d.shape), 6)).clip(0, 1)
        base = base * (1 - yw[..., None]) + np.array([195, 180, 60.0]) * yw[..., None]
        for _ in range(16):
            px, py = rng.uniform(-0.7, 0.7) * a, rng.uniform(-0.6, 0.6) * b
            r = rng.uniform(3, 9)
            ex = cx + px * np.cos(ang) - py * np.sin(ang); ey = cy + px * np.sin(ang) + py * np.cos(ang)
            dist = np.hypot(xx - ex, yy - ey)
            base = np.where((dist < r * 1.9)[..., None], base * 0.5 + np.array([205, 180, 55.0]) * 0.5, base)
            base = np.where((dist < r)[..., None], np.array([112, 72, 38.0]), base)
        img[leaf] = base[leaf]
        img += rng.normal(0, 3, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)
    sev = _severity(size, rng, 4)
    stops = [0, .5, .62, .78, .88, 1.0]
    cols = np.array([[48, 125, 38], [66, 140, 40], [150, 172, 55], [205, 180, 60], [125, 82, 44], [105, 68, 38]], float)
    img = np.stack([np.interp(sev.ravel(), stops, cols[:, c]).reshape(sev.shape) for c in range(3)], -1)
    img *= (1 + 0.10 * np.sin(xx * 2 * np.pi / 9))[..., None]
    img += gaussian_filter(rng.normal(0, 14, img.shape), (0.8, 0.8, 0))
    soil = (np.abs(xx - size * 0.9) < 9) | (yy > size * 0.965)
    img[soil] = np.array([156, 118, 84.0]) + rng.normal(0, 6, (int(soil.sum()), 3))
    return np.clip(img, 0, 255).astype(np.uint8)
