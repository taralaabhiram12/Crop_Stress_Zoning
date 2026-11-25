"""Shared UI helpers."""
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from czs import BAND_NAMES, CLASS_COLORS, CLASS_NAMES
from czs.io_utils import load_cube
from czs.synthetic import generate_field

CSCALE = [[0, CLASS_COLORS[0]], [1 / 3, CLASS_COLORS[0]], [1 / 3, CLASS_COLORS[1]],
          [2 / 3, CLASS_COLORS[1]], [2 / 3, CLASS_COLORS[2]], [1, CLASS_COLORS[2]]]
STATUS = {"not_requested": "⚪ Not submitted", "pending": "🟡 Awaiting agronomist",
          "validated": "🟢 Validated", "resurvey": "🔴 Re-survey needed"}


def class_heatmap(z, title, boundary=None, overlay=None, height=420):
    from czs.validation import boundary_map  # noqa
    fig = go.Figure(go.Heatmap(
        z=z, zmin=0, zmax=2, colorscale=CSCALE, showscale=False,
        customdata=np.vectorize(lambda v: CLASS_NAMES[v])(z),
        hovertemplate="x=%{x}, y=%{y}<br>%{customdata}<extra></extra>"))
    if boundary is not None:
        yy, xx = np.nonzero(boundary)
        fig.add_trace(go.Scattergl(x=xx, y=yy, mode="markers", hoverinfo="skip",
                                   marker=dict(size=2, color="white")))
    if overlay is not None:
        fig.add_trace(go.Scatter(x=overlay[0], y=overlay[1], mode="lines", hoverinfo="skip",
                                 line=dict(color="black", width=3, dash="dash")))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), height=height, showlegend=False)
    return fig


def legend():
    st.markdown(" ".join(f"<span style='background:{c};color:#fff;padding:2px 10px;"
                         f"border-radius:10px;margin-right:6px'>{n}</span>"
                         for n, c in zip(CLASS_NAMES, CLASS_COLORS)), unsafe_allow_html=True)


def png_bytes(cls, scale=4):
    from PIL import Image
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in CLASS_COLORS], np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb[cls]).resize((cls.shape[1] * scale, cls.shape[0] * scale), Image.NEAREST).save(buf, "PNG")
    return buf.getvalue()


def zone_table(cls, area_ha=None):
    import pandas as pd
    rows = []
    for i in range(3):
        p = float((cls == i).mean() * 100)
        rows.append({"Zone": CLASS_NAMES[i], "Share %": round(p, 1),
                     "Area (ha)": round(area_ha * p / 100, 2) if area_ha else None})
    return pd.DataFrame(rows)


def zone_metrics(cls, area_ha=None):
    cols = st.columns(3)
    for i, c in enumerate(cols):
        p = (cls == i).mean() * 100
        c.metric(CLASS_NAMES[i], f"{p:.1f}%", f"{area_ha * p / 100:.1f} ha" if area_ha else None, delta_color="off")


@st.cache_data(show_spinner=False)
def gen_field(size, seed, patches, noise=0.015, unc=2.0):
    return generate_field(size, int(seed), patches, noise, unc)


def bands_input(prefix):
    """Imagery source widgets. Returns (bands_dict | None, source_label)."""
    src = st.radio("Imagery source", ["Simulated drone survey (demo)", "Upload multispectral raster"],
                   horizontal=True, key=f"{prefix}_src")
    if src.startswith("Sim"):
        c1, c2, c3 = st.columns(3)
        size = c1.select_slider("Resolution (px)", [64, 96, 128, 160], 128, key=f"{prefix}_size")
        seed = c2.number_input("Survey seed", 0, 9999, 42, key=f"{prefix}_seed",
                               help="Change to simulate a different flight / field condition.")
        patches = c3.slider("Stress patches", 2, 9, 5, key=f"{prefix}_patches")
        return gen_field(size, seed, patches)["bands"], "simulated"
    up = st.file_uploader("Multispectral raster (.tif / .npy / .npz)", type=["tif", "tiff", "npy", "npz"],
                          key=f"{prefix}_up")
    if up is None:
        st.info("Upload a multi-band image, or use the simulated survey.")
        return None, "upload"
    try:
        cube = load_cube(up.getvalue(), up.name)
    except Exception as e:  # noqa
        st.error(f"Could not read file: {e}")
        return None, "upload"
    st.success(f"{cube.shape[0]}×{cube.shape[1]} px, {cube.shape[2]} bands")
    cols = st.columns(5)
    order = {n: cols[i].selectbox(n.upper(), range(cube.shape[2]), index=min(i, cube.shape[2] - 1),
                                  format_func=lambda j: f"Band {j + 1}", key=f"{prefix}_b_{n}")
             for i, n in enumerate(BAND_NAMES)}
    scale = st.number_input("Divide values by (e.g. 10000 for DN)", 1.0, 65535.0, 1.0, key=f"{prefix}_scale")
    return {n: cube[..., j] / scale for n, j in order.items()}, "upload"
