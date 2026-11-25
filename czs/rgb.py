"""Crop-stress analysis from ordinary RGB photos (phone / drone).

Pipeline: load -> canopy (plant) mask -> visible-band indices (ExG, VARI, GLI, NGRDI, TGI)
-> K-Means on colour/index features -> each cluster is classified Healthy / Stressed / Diseased
from its *physical* centroid colour (hue, greenness) so a healthy photo is never forced into
stress classes -> lesion statistics, hotspot grid, health score.
"""
import io

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import CLASS_COLORS

MAX_SIDE = 640
BG = 3  # zone value for "not vegetation"
GRADES = [(85, "Excellent"), (70, "Good"), (50, "Needs attention"), (30, "Poor"), (0, "Critical")]


def load_image(file_bytes, max_side=MAX_SIDE):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def rgb_to_hsv(f):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx, mn = f.max(-1), f.min(-1); d = mx - mn
    h = np.zeros_like(mx); m = d > 1e-6
    rc = (mx == r) & m; gc = (mx == g) & m & ~rc; bc = (mx == b) & m & ~rc & ~gc
    h[rc] = ((g - b)[rc] / d[rc]) % 6
    h[gc] = (b - r)[gc] / d[gc] + 2
    h[bc] = (r - g)[bc] / d[bc] + 4
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h * 60.0, s, mx


def vis_indices(f):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    t = r + g + b + 1e-6; rn, gn, bn = r / t, g / t, b / t
    exg = 2 * gn - rn - bn
    return {"ExG": exg, "ExGR": exg - (1.4 * rn - gn),
            "VARI": np.clip((g - r) / (g + r - b + 1e-6), -1, 1),
            "GLI": (2 * g - r - b) / (2 * g + r + b + 1e-6),
            "NGRDI": (g - r) / (g + r + 1e-6),
            "TGI": g - 0.39 * r - 0.61 * b}


def _otsu(x, bins=128):
    hist, edges = np.histogram(x, bins=bins)
    c = (edges[:-1] + edges[1:]) / 2; w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * c) / np.maximum(w0, 1); m1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def canopy_mask(rgb, mode="aerial", sensitivity=0.5, brown="auto"):
    f = rgb / 255.0; h, s, v = rgb_to_hsv(f); idx = vis_indices(f)
    if mode == "closeup":
        m = (s > 0.15) & (v > 0.10) & (v < 0.98)
    else:
        exg = idx["ExG"]; thr = _otsu(exg.ravel()) - (sensitivity - 0.5) * 0.12
        yellow = (h >= 45) & (h < 85) & (s > 0.35) & (v > 0.30)
        m = (exg > thr) | yellow
        # brown ground: bare soil or dead crop? colour alone cannot tell, so use context.
        # auto = in a dense canopy (>=55% green/yellow) large brown areas are dead crop; thin tracks are soil.
        if brown == "crop" or (brown == "auto" and m.mean() >= 0.55):
            cand = (((h < 45) | (h >= 300)) & (s > 0.28) & (v > 0.18)) & ~m
            r = 2 * max(4, int(0.04 * min(m.shape))) + 1
            cand = ndi.maximum_filter(ndi.minimum_filter(cand.astype(np.uint8), size=r), size=r).astype(bool)
            m = m | cand
        holes = ndi.binary_fill_holes(m) & ~m                  # enclosed dead patches stay canopy
        lab, n = ndi.label(holes)
        if n:
            area = ndi.sum(holes, lab, range(1, n + 1))
            m |= np.isin(lab, 1 + np.nonzero(area < 0.02 * m.size)[0])
    m = ndi.binary_opening(m, iterations=1)
    lab, n = ndi.label(m)
    if n:
        area = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.nonzero(area >= 25)[0])
    return m


def classify_color(hue, gli):
    """Physical rule used to name a cluster from its centroid colour."""
    if 85 <= hue <= 175:
        return 0 if gli > 0.03 else 1
    if 45 <= hue < 85:
        return 1          # yellow / yellow-green: chlorosis
    if hue < 45:
        return 2          # brown / orange / red: necrosis
    return 1


def grade(score):
    return next(name for t, name in GRADES if score >= t)


def analyse(rgb, mode="aerial", k=5, sensitivity=0.5, seed=0, grid=3, brown="auto"):
    f = rgb / 255.0; h, s, v = rgb_to_hsv(f); idx = vis_indices(f)
    mask = canopy_mask(rgb, mode, sensitivity, brown)
    zone = np.full(mask.shape, BG, np.uint8)
    n = int(mask.sum())
    if n < 100:
        raise ValueError("Very little vegetation was detected. Try the other image type or raise the sensitivity.")
    hs = np.where(h >= 300, h - 360, h)
    F = np.stack([idx["GLI"], idx["VARI"], (np.clip(hs, -60, 180) + 60) / 240, s, v, idx["ExG"]], -1)[mask]
    Xs = StandardScaler().fit_transform(F)
    rng = np.random.default_rng(seed)
    fit = Xs[rng.choice(n, min(n, 20000), replace=False)]
    km = KMeans(n_clusters=int(min(k, 8)), n_init=5, random_state=seed).fit(fit)
    lab = km.predict(Xs)
    hue_m, gli_m = hs[mask], idx["GLI"][mask]
    cmap = np.zeros(km.n_clusters, int); centroid = []
    for c in range(km.n_clusters):
        sel = lab == c
        if not sel.any():
            continue
        ch, cg = float(np.median(hue_m[sel])), float(np.median(gli_m[sel]))
        cmap[c] = classify_color(ch, cg); centroid.append((c, int(sel.sum()), ch, cg, int(cmap[c])))
    zone[mask] = cmap[lab].astype(np.uint8)
    inner = ndi.median_filter(zone, size=3)                    # remove salt-and-pepper
    zone = np.where(mask & (inner != BG), inner, zone).astype(np.uint8)
    zone[~mask] = BG

    cnt = [(zone == i).sum() for i in range(3)]; tot = max(sum(cnt), 1)
    pct = [100.0 * float(c) / tot for c in cnt]
    score = pct[0] + 0.5 * pct[1]
    dis = zone == 2; lab_d, nl = ndi.label(dis, structure=np.ones((3, 3)))
    areas = ndi.sum(dis, lab_d, range(1, nl + 1)) if nl else np.array([])
    min_px = max(6, int(0.0003 * tot)); big = areas[areas >= min_px] if nl else areas
    H, W = zone.shape; gs = []
    for gi in range(grid):
        row = []
        for gj in range(grid):
            cell = zone[gi * H // grid:(gi + 1) * H // grid, gj * W // grid:(gj + 1) * W // grid]
            cc = [(cell == i).sum() for i in range(3)]; t = sum(cc)
            row.append(None if t < 0.05 * cell.size else round(100.0 * (cc[0] + 0.5 * cc[1]) / t, 1))
        gs.append(row)
    details = dict(mode=mode, size=[W, H], canopy_pct=round(100.0 * float((zone != BG).mean()), 1),
                   zone_pct=[float(round(p, 1)) for p in pct], health_score=float(round(score, 1)), grade=grade(score),
                   lesion_count=int(len(big)),
                   lesion_median_pct=round(100.0 * float(np.median(big)) / tot, 2) if len(big) else 0.0,
                   largest_patch_pct=round(100.0 * float(areas.max()) / tot, 1) if nl else 0.0,
                   mean_exg=round(float(idx["ExG"][mask].mean()), 3), mean_vari=round(float(idx["VARI"][mask].mean()), 3),
                   mean_gli=round(float(idx["GLI"][mask].mean()), 3), mean_hue=round(float(np.median(hs[mask])), 1),
                   grid=gs, clusters=[dict(id=c[0], px=c[1], hue=round(c[2], 1), gli=round(c[3], 3), cls=c[4]) for c in centroid])
    return dict(zone=zone, mask=mask, indices=idx, hue=hs, details=details)


def overlay(rgb, zone, alpha=0.55):
    cols = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in CLASS_COLORS], float)
    out = rgb.astype(float).copy()
    for i in range(3):
        m = zone == i
        out[m] = (1 - alpha) * out[m] + alpha * cols[i]
    bg = zone == BG
    out[bg] = 0.75 * out[bg] + 0.25 * 128
    return out.astype(np.uint8)


def advise_rgb(d):
    h, s, dz = d["zone_pct"]; out = []
    if d["health_score"] >= 85 and dz < 5:
        out.append(("Low", "Crop looks healthy", "Colour analysis shows mostly vigorous green canopy. Continue routine monitoring and re-scan in 7–10 days."))
    if dz >= 20:
        out.append(("High", "Large necrotic / diseased area", f"{dz:.0f}% of the visible canopy is brown or dead. Scout these areas within 48 hours, collect leaf samples and consult an agronomist before treating."))
    elif dz >= 5:
        out.append(("Medium", "Browning detected", f"{dz:.0f}% of the canopy shows browning. Inspect affected leaves and neighbouring plants this week."))
    if d["lesion_count"] >= 12 and d["lesion_median_pct"] < 0.8:
        out.append(("Medium", "Scattered spot pattern", f"{d['lesion_count']} separate small lesions were found. This pattern is typical of leaf-spot or blight-type problems; inspect the underside of leaves and check humidity and recent rain."))
    if d["largest_patch_pct"] >= 8:
        out.append(("High", "Large contiguous dead patch", f"One connected patch covers {d['largest_patch_pct']:.0f}% of the canopy. Large patches often point to root, soil, irrigation or pest-outbreak problems rather than leaf disease."))
    if s >= 30 and dz < 15:
        out.append(("Medium", "Widespread yellowing (chlorosis)", f"{s:.0f}% of the canopy is yellow/yellow-green. Common causes are nitrogen deficiency, waterlogging or early stress; consider soil or tissue testing and check drainage."))
    elif s >= 12:
        out.append(("Low", "Mild yellowing", f"{s:.0f}% of the canopy shows early yellowing. Monitor and compare with the next scan."))
    if d["canopy_pct"] < 30 and d["mode"] == "aerial":
        out.append(("Low", "Low canopy cover", "Little vegetation was detected, which may be an early growth stage or bare soil in view."))
    if not out:
        out.append(("Low", "Routine monitoring", "No strong stress signal. Re-scan after a week."))
    return out


DISCLAIMER = ("Colour-based screening from a photo. It cannot identify the exact disease or deficiency. "
              "Confirm in the field or with an agronomist before treating.")
