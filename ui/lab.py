"""Analysis Lab: full-control K-Means exploration (original standalone dashboard)."""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from czs import BAND_NAMES, CLASS_COLORS, CLASS_NAMES
from czs.clustering import build_features, k_selection, kmeans_zones
from czs.indices import INDEX_INFO, compute_indices
from czs.io_utils import load_cube, load_mask
from czs.synthetic import generate_field
from czs.validation import boundary_f1, boundary_map, metrics
from ui.components import header
from ui.common import class_heatmap, legend, png_bytes


def render_lab():
    header("🔬 Analysis Lab")
    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("1 · Data")
        src = st.radio("Source", ["Synthetic demo field", "Upload imagery"], horizontal=True)
        bands = truth = None
        if src == "Synthetic demo field":
            size = st.slider("Field size (px)", 64, 192, 128, 16)
            seed = st.number_input("Random seed", 0, 9999, 42)
            patches = st.slider("Stress patches", 2, 9, 5)
            noise = st.slider("Sensor noise", 0.0, 0.05, 0.015, 0.005)
            unc = st.slider("Agronomist boundary uncertainty", 0.0, 5.0, 2.0, 0.5,
                            help="How far hand-drawn boundaries drift from the real ones.")

            @st.cache_data(show_spinner=False)
            def _gen(*a):
                return generate_field(*a)
            d = _gen(size, int(seed), patches, noise, unc)
            bands, truth = d["bands"], d["agronomist"]
        else:
            up = st.file_uploader("Multispectral raster (.tif / .npy / .npz)", type=["tif", "tiff", "npy", "npz"])
            if up is not None:
                try:
                    cube = load_cube(up.getvalue(), up.name)
                    st.success(f"{cube.shape[0]}×{cube.shape[1]} px, {cube.shape[2]} bands")
                    order = {}
                    for i, n in enumerate(BAND_NAMES):
                        order[n] = st.selectbox(f"Band for {n.upper()}", range(cube.shape[2]),
                                                index=min(i, cube.shape[2] - 1), format_func=lambda j: f"Band {j + 1}")
                    scale = st.number_input("Divide values by (e.g. 10000 for DN)", 1.0, 65535.0, 1.0)
                    bands = {n: cube[..., j] / scale for n, j in order.items()}
                    m = st.file_uploader("Agronomist mask (.png / .npy: 0/1/2)", type=["png", "jpg", "npy"])
                    if m is not None:
                        truth = load_mask(m.getvalue(), m.name, cube.shape[:2])
                except Exception as e:
                    st.error(f"Could not read file: {e}")
            else:
                st.info("Upload a file, or switch to the synthetic demo.")

        st.header("2 · Features")
        all_idx = list(INDEX_INFO)
        idx_sel = st.multiselect("Vegetation indices", all_idx, default=["NDVI", "GNDVI", "NDRE", "SAVI"],
                                 help="The first selected index orders clusters by health.")
        band_sel = st.multiselect("Raw bands (optional)", BAND_NAMES)

        st.header("3 · K-Means")
        k = st.slider("Clusters (k)", 2, 8, 3)
        smooth = st.select_slider("Spatial smoothing (median window)", [0, 3, 5, 7, 9], value=3)
        seed_k = st.number_input("K-Means seed", 0, 999, 0)
        tol = st.slider("Boundary tolerance (px)", 1, 10, 3)

    if bands is None or not idx_sel:
        st.warning("Provide imagery and select at least one index to begin.")
        st.stop()

    # ---------------------------------------------------------------- compute
    indices = compute_indices(bands)
    X, fnames, shape = build_features(indices, bands, idx_sel, band_sel)
    cls, cid, cls_of_cluster, km = kmeans_zones(X, shape, k=k, ref_col=0, seed=int(seed_k), smooth=smooth)

    tab1, tab2, tab3, tab4 = st.tabs(["🛰️ Imagery & Indices", "🧩 Stress Zones", "✅ Agronomist Validation", "📦 Export"])

    # ---- Tab 1
    with tab1:
        c1, c2 = st.columns(2)
        comp = np.stack([bands["nir"], bands["red"], bands["green"]], -1)
        comp = np.clip(comp / np.percentile(comp, 99), 0, 1)
        f1 = go.Figure(go.Image(z=(comp * 255).astype(np.uint8)))
        f1.update_layout(title="False-colour composite (NIR-R-G)", margin=dict(l=0, r=0, t=40, b=0), height=420)
        f1.update_xaxes(visible=False); f1.update_yaxes(visible=False)
        c1.plotly_chart(f1, width="stretch")
        show = c2.selectbox("Index map", list(INDEX_INFO), index=0)
        f2 = go.Figure(go.Heatmap(z=indices[show], colorscale="RdYlGn", colorbar=dict(title=show)))
        f2.update_yaxes(autorange="reversed", scaleanchor="x", visible=False); f2.update_xaxes(visible=False)
        f2.update_layout(title=f"{show} = {INDEX_INFO[show][0]}", margin=dict(l=0, r=0, t=40, b=0), height=420)
        c2.plotly_chart(f2, width="stretch")
        c2.caption(INDEX_INFO[show][1])
        st.subheader("Index statistics")
        st.dataframe(pd.DataFrame({n: pd.Series(v.ravel()).describe() for n, v in indices.items()}).T.round(3),
                     width="stretch")
        st.subheader("Spectral signature by zone")
        sig = go.Figure()
        for c in range(3):
            if (cls == c).any():
                sig.add_trace(go.Scatter(x=[b.upper() for b in BAND_NAMES],
                                         y=[float(bands[b][cls == c].mean()) for b in BAND_NAMES],
                                         mode="lines+markers", name=CLASS_NAMES[c],
                                         line=dict(color=CLASS_COLORS[c])))
        sig.update_layout(yaxis_title="Mean reflectance", height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(sig, width="stretch")

    # ---- Tab 2
    with tab2:
        legend()
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(class_heatmap(cls, f"K-Means stress zones (k={k})", boundary_map(cls)),
                        width="stretch")
        area = pd.DataFrame({"Zone": CLASS_NAMES, "Pixels": [(cls == i).sum() for i in range(3)]})
        area["Area %"] = (area["Pixels"] / cls.size * 100).round(1)
        with c2:
            st.subheader("Zone area")
            st.dataframe(area, hide_index=True, width="stretch")
            pie = go.Figure(go.Pie(labels=CLASS_NAMES, values=area["Pixels"], hole=.5,
                                   marker=dict(colors=CLASS_COLORS), sort=False))
            pie.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(pie, width="stretch")
        st.subheader("Cluster centroids (original feature units)")
        rows = []
        for c in range(k):
            sel = km.labels_ == c
            rows.append({"Cluster": c, "Zone": CLASS_NAMES[cls_of_cluster[c]], "Pixels": int(sel.sum()),
                         **{n: round(float(X[sel, j].mean()), 3) for j, n in enumerate(fnames)}})
        st.dataframe(pd.DataFrame(rows).sort_values(fnames[0], ascending=False), hide_index=True,
                     width="stretch")

        st.subheader("Choosing k")
        if st.button("Run elbow / silhouette analysis"):
            with st.spinner("Clustering for k = 2…8"):
                res = k_selection(X)
            ks = [r[0] for r in res]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=ks, y=[r[1] for r in res], name="Inertia (elbow)", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=ks, y=[r[2] for r in res], name="Silhouette", mode="lines+markers"),
                          secondary_y=True)
            fig.update_xaxes(title="k"); fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
            st.success(f"Best silhouette at k = {ks[int(np.argmax([r[2] for r in res]))]}")

        st.subheader("Pixel inspector")
        px, py = st.columns(2)
        x0 = px.slider("x", 0, shape[1] - 1, shape[1] // 2); y0 = py.slider("y", 0, shape[0] - 1, shape[0] // 2)
        st.write(f"Pixel ({x0}, {y0}) → **{CLASS_NAMES[cls[y0, x0]]}** (cluster {cid[y0, x0]}) · " +
                 " · ".join(f"{n} {indices[n][y0, x0]:.3f}" for n in idx_sel))

    # ---- Tab 3
    with tab3:
        if truth is None:
            st.info("Upload an agronomist mask in the sidebar to enable validation.")
        else:
            m = metrics(cls, truth); bf = boundary_f1(cls, truth, tol)
            a, b, c, d, e = st.columns(5)
            a.metric("Overall accuracy", f"{m['overall_accuracy'] * 100:.1f}%")
            b.metric("Cohen's κ", f"{m['kappa']:.3f}")
            c.metric("Mean IoU", f"{m['miou']:.3f}")
            d.metric(f"Boundary F1 (±{tol}px)", f"{bf['f1']:.3f}")
            e.metric("Boundary recall", f"{bf['recall']:.3f}")
            legend()
            l, r = st.columns(2)
            l.plotly_chart(class_heatmap(truth, "Agronomist markings (ground truth)", boundary_map(truth)),
                           width="stretch")
            r.plotly_chart(class_heatmap(cls, "K-Means zones", boundary_map(cls)), width="stretch")

            l, r = st.columns(2)
            agree = (cls == truth)
            ov = go.Figure(go.Heatmap(z=agree.astype(int), colorscale=[[0, "#d32f2f"], [1, "#e8f5e9"]],
                                      showscale=False, hovertemplate="%{z}<extra>1 = agree</extra>"))
            ov.update_yaxes(autorange="reversed", scaleanchor="x", visible=False); ov.update_xaxes(visible=False)
            ov.update_layout(title="Agreement map (red = disagreement)", height=380, margin=dict(l=0, r=0, t=40, b=0))
            l.plotly_chart(ov, width="stretch")
            cm = m["cm"]; cmn = cm / cm.sum(1, keepdims=True).clip(1)
            hm = go.Figure(go.Heatmap(z=cmn, x=CLASS_NAMES, y=CLASS_NAMES, colorscale="Blues", zmin=0, zmax=1,
                                      text=[[f"{cm[i, j]}<br>{cmn[i, j]:.0%}" for j in range(3)] for i in range(3)],
                                      texttemplate="%{text}"))
            hm.update_layout(title="Confusion matrix (row-normalised)", xaxis_title="K-Means",
                             yaxis_title="Agronomist", height=380, margin=dict(l=0, r=0, t=40, b=0))
            hm.update_yaxes(autorange="reversed")
            r.plotly_chart(hm, width="stretch")
            st.subheader("Per-zone accuracy")
            st.dataframe(pd.DataFrame([{"Zone": CLASS_NAMES[p["cls"]], "Precision": round(p["precision"], 3),
                                        "Recall": round(p["recall"], 3), "F1": round(p["f1"], 3),
                                        "IoU": round(p["iou"], 3)} for p in m["per_class"]]),
                         hide_index=True, width="stretch")
            verdict = ("✅ Zones are in strong agreement with the agronomist." if m["kappa"] > 0.8 else
                       "🟡 Substantial agreement — review disagreement areas on the map." if m["kappa"] > 0.6 else
                       "🔴 Weak agreement — try other indices, a different k, or more smoothing.")
            st.markdown(f"**Verdict:** {verdict}")

    # ---- Tab 4
    with tab4:
        st.download_button("⬇️ Zone map (PNG)", png_bytes(cls), "stress_zones.png", "image/png")
        buf = io.BytesIO(); np.save(buf, cls)
        st.download_button("⬇️ Zone raster (.npy)", buf.getvalue(), "stress_zones.npy")
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "zone": [CLASS_NAMES[v] for v in cls.ravel()],
                           **{n: indices[n].ravel() for n in idx_sel}})
        if truth is not None:
            df["agronomist"] = [CLASS_NAMES[v] for v in truth.ravel()]
        st.download_button("⬇️ Per-pixel table (CSV)", df.to_csv(index=False), "pixels.csv", "text/csv")
        st.dataframe(df.head(200), width="stretch")
