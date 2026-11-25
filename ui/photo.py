"""Photo (RGB) crop-stress analysis for farmers and agronomists."""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from czs import CLASS_COLORS, CLASS_NAMES, db
from czs.report import build_photo_report
from czs.rgb import BG, DISCLAIMER, advise_rgb, analyse, classify_color, load_image, overlay
from czs.synthetic import rgb_demo
from ui.common import STATUS
from ui.components import comment_thread, gauge, grade_badge, header

AERIAL, CLOSE = "🚁 Field / drone view", "🍃 Close-up leaf or canopy"
PHOTO_STATUS = {"not_requested": "⚪ Not submitted", "pending": "🟡 Awaiting agronomist", "reviewed": "🟢 Reviewed"}
CELL = "ABCDEFGH"


@st.cache_data(show_spinner=False)
def _demo_bytes(kind):
    buf = io.BytesIO(); Image.fromarray(rgb_demo(kind)).save(buf, "PNG"); return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=40)
def _analyse(data, mode, k, sens, grid, brown):
    rgb = load_image(data)
    return rgb, analyse(rgb, mode, k, sens, 0, grid, brown)


def _img(arr, title=None, height=None):
    fig = go.Figure(go.Image(z=arr))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=30 if title else 0, b=0), title=title, height=height)
    return fig


def _png(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG"); return buf.getvalue()


def _grid_fig(res, rgb, alpha):
    d = res["details"]; g = d["grid"]; n = len(g); H, W = res["zone"].shape
    fig = _img(overlay(rgb, res["zone"], alpha))
    for i in range(1, n):
        fig.add_shape(type="line", x0=i * W / n, x1=i * W / n, y0=0, y1=H, line=dict(color="white", width=2, dash="dash"))
        fig.add_shape(type="line", y0=i * H / n, y1=i * H / n, x0=0, x1=W, line=dict(color="white", width=2, dash="dash"))
    for i in range(n):
        for j in range(n):
            s = g[i][j]
            fig.add_annotation(x=(j + .5) * W / n, y=(i + .5) * H / n, showarrow=False,
                               text=f"<b>{CELL[i]}{j + 1}</b><br>{'–' if s is None else f'{s:.0f}'}",
                               font=dict(color="white", size=13), bgcolor="rgba(0,0,0,.45)")
    return fig


def _hue_fig(res):
    m = res["mask"]; hue = res["hue"][m]
    cnt, edges = np.histogram(hue, bins=np.arange(-60, 185, 5)); c = (edges[:-1] + edges[1:]) / 2
    cols = [CLASS_COLORS[classify_color(x, 0.1)] for x in c]
    fig = go.Figure(go.Bar(x=c, y=100 * cnt / max(cnt.sum(), 1), marker_color=cols,
                           hovertemplate="hue %{x}°<br>%{y:.1f}% of canopy<extra></extra>"))
    fig.add_vrect(x0=85, x1=175, fillcolor="#2e7d32", opacity=.07, line_width=0, annotation_text="healthy green")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Hue (°)  ·  brown/red → yellow → green",
                      yaxis_title="% of canopy", bargap=.05)
    return fig


def _metrics(d, area_ha=None):
    z = d["zone_pct"]
    r1 = st.columns(3)
    r1[0].metric("🟢 Healthy", f"{z[0]:.1f}%"); r1[1].metric("🟡 Yellowing", f"{z[1]:.1f}%"); r1[2].metric("🔴 Browning / diseased", f"{z[2]:.1f}%")
    r2 = st.columns(3)
    r2[0].metric("🌿 Canopy cover", f"{d['canopy_pct']:.0f}%"); r2[1].metric("🔬 Lesions found", d["lesion_count"])
    r2[2].metric("🧱 Largest dead patch", f"{d['largest_patch_pct']:.1f}%")


def _findings(d):
    for prio, title, text in advise_rgb(d):
        {"High": st.error, "Medium": st.warning, "Low": st.success}[prio](f"**{title}**  \n{text}")
    st.caption(DISCLAIMER)


def _worst_cells(d, n=3):
    cells = [(d["grid"][i][j], f"{CELL[i]}{j + 1}") for i in range(len(d["grid"])) for j in range(len(d["grid"])) if d["grid"][i][j] is not None]
    return sorted(cells)[:n]


# ------------------------------------------------------------------ analyse tab
def _analyze_tab(user):
    c1, c2 = st.columns([2, 1])
    files = c1.file_uploader("Upload crop photos (JPG / PNG / WEBP, up to 10)", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True, key="ph_files")
    with c1:
        b1, b2, b3 = st.columns(3)
        b1.button("🚁 Try sample field photo", width="stretch",
                  on_click=lambda: st.session_state.update(ph_demo="aerial", ph_mode=AERIAL))
        b2.button("🍃 Try sample leaf photo", width="stretch",
                  on_click=lambda: st.session_state.update(ph_demo="leaf", ph_mode=CLOSE))
        b3.button("✖ Clear sample", width="stretch", on_click=lambda: st.session_state.pop("ph_demo", None))
    with c2:
        mode_l = st.radio("Photo type", [AERIAL, CLOSE], key="ph_mode",
                          help="Field view removes soil; close-up treats everything coloured as plant.")
        mode = "aerial" if mode_l == AERIAL else "closeup"
        sens = st.slider("Plant detection sensitivity", 0.0, 1.0, 0.5, 0.05, disabled=mode == "closeup")
        brown_l = st.selectbox("Brown areas are…", ["Auto (recommended)", "Dead / diseased crop", "Bare soil (ignore)"], disabled=mode == "closeup",
                               help="Colour alone cannot tell dead crop from soil. Auto counts large brown areas as dead crop when the field is densely planted, and ignores thin soil tracks.")
        brown = {"Auto (recommended)": "auto", "Dead / diseased crop": "crop", "Bare soil (ignore)": "soil"}[brown_l]
        with st.expander("⚙️ Advanced"):
            k = st.slider("K-Means clusters", 3, 8, 5); gridn = st.select_slider("Hotspot grid", [2, 3, 4, 5], 3)
    if "k" not in locals(): k, gridn = 5, 3

    items = [(f.name, f.getvalue()) for f in files[:10]]
    if not items and st.session_state.get("ph_demo"):
        kind = st.session_state["ph_demo"]; items = [(f"sample_{kind}.png", _demo_bytes(kind))]
    if not items:
        st.info("📸 Upload one or more photos (or try a sample) to see health score, yellowing and browning share, "
                "lesion count and hotspot areas. Tip: take photos in daylight, from the same height each time.")
        return
    results = {}
    with st.spinner("Analysing photos…"):
        for name, data in items:
            try:
                rgb, res = _analyse(data, mode, k, sens, gridn, brown); results[name] = (rgb, res)
            except Exception as e:  # noqa
                st.warning(f"**{name}**: {e}")
    if not results:
        return
    if len(results) > 1:
        df = pd.DataFrame([{"Photo": n, "Health score": r["details"]["health_score"], "Grade": r["details"]["grade"],
                            "Healthy %": r["details"]["zone_pct"][0], "Yellowing %": r["details"]["zone_pct"][1],
                            "Browning %": r["details"]["zone_pct"][2], "Lesions": r["details"]["lesion_count"],
                            "Canopy %": r["details"]["canopy_pct"]} for n, (_, r) in results.items()])
        st.subheader("Batch summary"); st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("⬇️ Summary (CSV)", df.to_csv(index=False), "photo_summary.csv", "text/csv")
    name = st.selectbox("Photo", list(results)) if len(results) > 1 else next(iter(results))
    rgb, res = results[name]; d = res["details"]
    st.divider()
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(gauge(d["health_score"]), width="stretch", key="gauge_main"); grade_badge(d["grade"])
    with g2:
        _metrics(d)
    t1, t2, t3, t4, t5 = st.tabs(["🖼️ Visual", "🧭 Hotspot grid", "🎨 Colour evidence", "💡 Findings", "💾 Save & export"])
    with t1:
        alpha = st.slider("Overlay strength", 0.1, 0.9, 0.55, 0.05)
        a, b = st.columns(2)
        a.plotly_chart(_img(rgb, "Original"), width="stretch", key="img_o")
        view = b.radio("Right panel", ["Stress zones", "Greenness map", "Plant mask"], horizontal=True, label_visibility="collapsed")
        if view == "Stress zones":
            b.plotly_chart(_img(overlay(rgb, res["zone"], alpha), "Stress zones"), width="stretch", key="img_z")
        elif view == "Greenness map":
            z = np.where(res["mask"], res["indices"]["GLI"], np.nan)
            f = go.Figure(go.Heatmap(z=z, colorscale="RdYlGn", zmin=-0.1, zmax=0.35, colorbar=dict(title="GLI")))
            f.update_yaxes(autorange="reversed", scaleanchor="x", visible=False); f.update_xaxes(visible=False)
            f.update_layout(title="Greenness (GLI)", margin=dict(l=0, r=0, t=30, b=0)); b.plotly_chart(f, width="stretch", key="img_g")
        else:
            b.plotly_chart(_img((res["mask"] * 255).astype(np.uint8), "Detected plant pixels"), width="stretch", key="img_m")
        st.markdown(" ".join(f"<span style='background:{c};color:#fff;padding:2px 10px;border-radius:10px;margin-right:6px'>{n}</span>"
                             for n, c in zip(["Healthy", "Yellowing", "Browning / diseased"], CLASS_COLORS)), unsafe_allow_html=True)
    with t2:
        st.caption("Scores are crop health (0–100) per cell. Scout the lowest cells first.")
        a, b = st.columns([3, 2])
        a.plotly_chart(_grid_fig(res, rgb, 0.45), width="stretch", key="grid_img")
        with b:
            n = len(d["grid"])
            z = [[np.nan if v is None else v for v in row] for row in d["grid"]]
            hm = go.Figure(go.Heatmap(z=z, x=[str(j + 1) for j in range(n)], y=list(CELL[:n]), colorscale="RdYlGn", zmin=0, zmax=100,
                                      text=[["–" if v is None else f"{v:.0f}" for v in row] for row in d["grid"]], texttemplate="%{text}"))
            hm.update_yaxes(autorange="reversed"); hm.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(hm, width="stretch", key="grid_hm")
            st.markdown("**Scout first:** " + ", ".join(f"**{c}** ({s:.0f})" for s, c in _worst_cells(d)))
    with t3:
        st.plotly_chart(_hue_fig(res), width="stretch", key="hue")
        cl = pd.DataFrame([{"Cluster": c["id"], "Share of canopy %": round(100 * c["px"] / max(sum(x["px"] for x in d["clusters"]), 1), 1),
                            "Median hue °": c["hue"], "Greenness (GLI)": c["gli"], "Zone": ["Healthy", "Yellowing", "Browning / diseased"][c["cls"]]}
                           for c in d["clusters"]])
        st.caption("K-Means groups similar pixel colours; each cluster is named from its real colour, so a healthy photo is never forced into stress classes.")
        st.dataframe(cl, hide_index=True, width="stretch")
        st.write(f"Mean ExG **{d['mean_exg']}** · VARI **{d['mean_vari']}** · GLI **{d['mean_gli']}** · median hue **{d['mean_hue']}°**")
    with t4:
        _findings(d)
    with t5:
        ov = overlay(rgb, res["zone"], 0.55)
        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇️ Annotated image (PNG)", _png(ov), f"{name}_zones.png", "image/png", width="stretch")
        rep = build_photo_report(d, rgb, ov, advise_rgb(d), DISCLAIMER, f"Crop photo analysis – {name}", f"Generated for {user['name']}")
        e2.download_button("⬇️ Report (HTML → PDF)", rep, f"{name}_report.html", "text/html", width="stretch")
        e3.download_button("⬇️ Details (JSON)", pd.Series(d).to_json(), f"{name}_details.json", width="stretch")
        st.markdown("#### Save to a field record")
        fields = db.list_fields(user["id"] if user["role"] == "farmer" else None)
        if not fields:
            st.info("Add a field first (🌱 My Fields) to save results.")
        else:
            fld = st.selectbox("Field", fields, format_func=lambda f: f"{f['name']} – {f['farmer_name']}" if user["role"] != "farmer" else f["name"], key="ph_field")
            note = st.text_input("Note (optional) – e.g. 'north corner, 3 days after rain'")
            req = st.checkbox("Ask an agronomist to review this photo", user["role"] == "farmer")
            if st.button("💾 Save photo analysis", type="primary"):
                pid = db.save_photo(fld["id"], rgb, res["zone"], d, note, req); st.toast(f"Saved as photo #{pid} ✅")


# ------------------------------------------------------------------ saved tab / viewer
def _viewer(p, user, reviewer=False):
    rgb, zone = db.load_photo_arrays(p["id"]); d = p["details"]
    st.caption(f"**{p['field_name']}** · {p['crop']} · farmer {p['farmer_name']} · {p['created'][:16].replace('T', ' ')} · {PHOTO_STATUS[p['review_status']]}")
    if p["note"]:
        st.write(f"📝 {p['note']}")
    g1, g2 = st.columns([1, 2])
    g1.plotly_chart(gauge(d["health_score"], height=220), width="stretch", key=f"pg_{p['id']}"); g2.empty()
    with g2:
        _metrics(d)
    a, b = st.columns(2)
    a.plotly_chart(_img(rgb, "Photo"), width="stretch", key=f"po_{p['id']}")
    b.plotly_chart(_img(overlay(rgb, zone, 0.55), "Stress zones"), width="stretch", key=f"pz_{p['id']}")
    with st.expander("💡 Automatic findings", expanded=not p["diagnosis"]):
        _findings(d)
    if p["diagnosis"]:
        st.success(f"**Agronomist diagnosis** ({p['reviewer_name']}, {str(p['reviewed'])[:10]})  \n{p['diagnosis']}")
    if reviewer:
        with st.form(f"dx_{p['id']}"):
            dx = st.text_area("Your diagnosis and advice", value=p["diagnosis"] or "", height=120)
            if st.form_submit_button("📨 Submit diagnosis", type="primary") and dx.strip():
                db.review_photo(p["id"], user["id"], dx.strip()); st.toast("Diagnosis sent ✅"); st.rerun()
    elif p["review_status"] == "not_requested" and user["role"] == "farmer":
        if st.button("📨 Ask an agronomist to review", key=f"rq_{p['id']}"):
            db.request_photo_review(p["id"]); st.rerun()
    st.markdown("##### 💬 Conversation"); comment_thread("photo", p["id"], user)
    ov = overlay(rgb, zone, 0.55)
    rep = build_photo_report(d, rgb, ov, advise_rgb(d), DISCLAIMER, f"Photo #{p['id']} – {p['field_name']}",
                             f"{p['farmer_name']} · {p['created'][:10]}", p["diagnosis"])
    st.download_button("⬇️ Report (HTML → PDF)", rep, f"photo_{p['id']}_report.html", "text/html", key=f"dl_{p['id']}")
    if user["role"] == "farmer" and st.button("🗑️ Delete this photo", key=f"del_{p['id']}"):
        db.delete_photo(p["id"]); st.rerun()


def _saved_tab(user):
    photos = db.list_photos(owner_id=user["id"] if user["role"] == "farmer" else None)
    if not photos:
        st.info("No saved photo analyses yet. Analyse a photo and press **Save**."); return
    if len(photos) > 1:
        df = pd.DataFrame([{"date": p["created"][:10], "field": p["field_name"], "score": p["details"]["health_score"]} for p in reversed(photos)])
        fig = go.Figure([go.Scatter(x=g["date"], y=g["score"], name=f, mode="lines+markers") for f, g in df.groupby("field")])
        fig.update_layout(title="Photo health score over time", height=280, yaxis=dict(range=[0, 100]), margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    p = st.selectbox("Photo", photos, format_func=lambda p: f"#{p['id']} · {p['field_name']} · {p['created'][:10]} · score {p['details']['health_score']:.0f} · {PHOTO_STATUS[p['review_status']]}")
    _viewer(p, user)


def render_photo(user):
    header("📷 Photo Analysis")
    t1, t2 = st.tabs(["🔍 Analyze photos", "🗂️ Saved photos"])
    with t1:
        _analyze_tab(user)
    with t2:
        _saved_tab(user)


def render_reviews(user):
    header("📷 Photo Reviews")
    photos = db.list_photos()
    pend = [p for p in photos if p["review_status"] == "pending"]
    c = st.columns(3); c[0].metric("Awaiting diagnosis", len(pend)); c[1].metric("Reviewed", sum(p["review_status"] == "reviewed" for p in photos)); c[2].metric("Total photos", len(photos))
    if not photos:
        st.info("No photos have been saved by farmers yet."); return
    only = st.toggle("Show only photos awaiting diagnosis", True)
    pool = pend if only else photos
    if not pool:
        st.success("All caught up. 🎉"); return
    pool = sorted(pool, key=lambda p: p["details"]["health_score"])
    p = st.selectbox("Photo (worst health first)", pool, format_func=lambda p: f"#{p['id']} · {p['field_name']} ({p['farmer_name']}) · score {p['details']['health_score']:.0f} · {PHOTO_STATUS[p['review_status']]}")
    _viewer(p, user, reviewer=True)
