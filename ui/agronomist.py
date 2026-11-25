"""Agronomist-facing pages: review queue, boundary validation, auto-tuning, analytics."""
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import median_filter

from czs import CLASS_COLORS, CLASS_NAMES, db
from czs.config import ALERT_DISEASED_PCT
from czs.io_utils import load_mask
from czs.pipeline import bands_from_cube, run_analysis
from czs.validation import boundary_f1, boundary_map, metrics
from ui.components import comment_thread, header
from ui.common import STATUS, class_heatmap, legend

PAGES = ["📋 Dashboard", "✅ Validate Scan", "📷 Photo Reviews", "🎯 Auto-Tune", "🗺️ All Fields", "📊 Analytics", "🔬 Analysis Lab"]


def _label(s):
    return (f"#{s['id']} · {s['field_name']} ({s['farmer_name']}) · {s['created'][:10]} · "
            f"{STATUS[s['review_status']]}")


def dashboard(user):
    header(f"Agronomist dashboard · {user['name']}")
    scans = db.list_scans()
    pend = [s for s in scans if s["review_status"] == "pending"]
    photos_pending = db.list_photos(status="pending")
    c = st.columns(5)
    c[4].metric("Photos awaiting diagnosis", len(photos_pending))
    c[0].metric("Pending scan reviews", len(pend)); c[1].metric("Validated", sum(s["review_status"] == "validated" for s in scans))
    c[2].metric("Re-survey flagged", sum(s["review_status"] == "resurvey" for s in scans))
    c[3].metric("Fields monitored", len(db.list_fields()))
    st.subheader("Multispectral review queue")
    if not pend:
        st.success("No scans are waiting for validation. 🎉")
    else:
        df = pd.DataFrame([{"Scan": s["id"], "Field": s["field_name"], "Farmer": s["farmer_name"], "Crop": s["crop"],
                            "Date": s["created"][:10], "Diseased %": round(s["stats"]["zone_pct"][2], 1),
                            "Priority": "🔴 High" if s["stats"]["zone_pct"][2] >= ALERT_DISEASED_PCT else "🟡 Normal"}
                           for s in pend]).sort_values("Diseased %", ascending=False)
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("Open **✅ Validate Scan** from the menu to review these.")


def _apply(mask, shape, cls, p):
    m = mask.copy(); H, W = m.shape
    if shape == "Rectangle":
        m[p[1]:p[3] + 1, p[0]:p[2] + 1] = cls
    else:
        yy, xx = np.ogrid[:H, :W]
        m[(yy - p[1]) ** 2 + (xx - p[0]) ** 2 <= p[2] ** 2] = cls
    return m


def validate(user):
    header("✅ Validate Scan")
    scans = db.list_scans()
    if not scans:
        st.info("No scans have been submitted yet."); return
    scans.sort(key=lambda s: (s["review_status"] != "pending", s["created"]), reverse=False)
    s = st.selectbox("Scan", scans, format_func=_label)
    sid = s["id"]
    cube, ai = db.load_scan_arrays(sid)
    H, W = ai.shape
    prev = db.get_review(sid)
    mk, hk = f"mask_{sid}", f"hist_{sid}"
    if mk not in st.session_state:
        st.session_state[mk] = (prev["mask"].copy() if prev and prev["mask"] is not None else ai.copy())
        st.session_state[hk] = []
    mask = st.session_state[mk]
    st.caption(f"Field **{s['field_name']}** · {s['crop']} · {s['area_ha'] or '–'} ha · farmer {s['farmer_name']} · "
               f"{STATUS[s['review_status']]}")
    legend()

    # ---- editing tools
    with st.expander("🖊️ Mark / correct zone boundaries", expanded=True):
        c1, c2 = st.columns([1, 2])
        shape = c1.radio("Tool", ["Rectangle", "Circle"], horizontal=True, key=f"sh_{sid}")
        cls_name = c1.selectbox("Assign zone", CLASS_NAMES, key=f"cl_{sid}")
        if shape == "Rectangle":
            xr = c2.slider("X range", 0, W - 1, (W // 4, W // 2), key=f"x_{sid}")
            yr = c2.slider("Y range", 0, H - 1, (H // 4, H // 2), key=f"y_{sid}")
            p = (xr[0], yr[0], xr[1], yr[1])
            ox = [p[0], p[2], p[2], p[0], p[0]]; oy = [p[1], p[1], p[3], p[3], p[1]]
        else:
            cx = c2.slider("Centre X", 0, W - 1, W // 2, key=f"cx_{sid}")
            cy = c2.slider("Centre Y", 0, H - 1, H // 2, key=f"cy_{sid}")
            r = c2.slider("Radius", 2, max(W, H) // 2, W // 8, key=f"r_{sid}")
            p = (cx, cy, r); t = np.linspace(0, 2 * np.pi, 60)
            ox, oy = cx + r * np.cos(t), cy + r * np.sin(t)
        b = st.columns(5)
        if b[0].button("Apply region", type="primary", key=f"ap_{sid}"):
            st.session_state[hk].append(mask.copy()); st.session_state[mk] = _apply(mask, shape, CLASS_NAMES.index(cls_name), p); st.rerun()
        if b[1].button("↩️ Undo", key=f"un_{sid}") and st.session_state[hk]:
            st.session_state[mk] = st.session_state[hk].pop(); st.rerun()
        if b[2].button("Smooth edges", key=f"sm_{sid}"):
            st.session_state[hk].append(mask.copy()); st.session_state[mk] = median_filter(mask, size=5).astype(np.uint8); st.rerun()
        if b[3].button("Reset to AI zones", key=f"rs_{sid}"):
            st.session_state[hk].append(mask.copy()); st.session_state[mk] = ai.copy(); st.rerun()
        up = b[4].file_uploader("Upload mask", type=["png", "npy"], key=f"upm_{sid}", label_visibility="collapsed")
        if up is not None and st.button("Use uploaded mask", key=f"um_{sid}"):
            st.session_state[hk].append(mask.copy()); st.session_state[mk] = load_mask(up.getvalue(), up.name, ai.shape); st.rerun()

    l, r = st.columns(2)
    l.plotly_chart(class_heatmap(ai, "K-Means zones (AI)", boundary_map(ai)), width="stretch")
    r.plotly_chart(class_heatmap(mask, "Agronomist zones (editable)", boundary_map(mask), overlay=(ox, oy)), width="stretch")

    m = metrics(ai, mask); bf = boundary_f1(ai, mask, 3)
    cc = st.columns(4)
    cc[0].metric("Agreement (accuracy)", f"{m['overall_accuracy'] * 100:.1f}%"); cc[1].metric("Cohen's κ", f"{m['kappa']:.3f}")
    cc[2].metric("Mean IoU", f"{m['miou']:.3f}"); cc[3].metric("Boundary F1 (±3px)", f"{bf['f1']:.3f}")
    cm = m["cm"]; cmn = cm / cm.sum(1, keepdims=True).clip(1)
    with st.expander("Confusion matrix & agreement map"):
        a, bb = st.columns(2)
        hm = go.Figure(go.Heatmap(z=cmn, x=CLASS_NAMES, y=CLASS_NAMES, colorscale="Blues", zmin=0, zmax=1,
                                  text=[[f"{cm[i, j]}" for j in range(3)] for i in range(3)], texttemplate="%{text}"))
        hm.update_layout(xaxis_title="K-Means", yaxis_title="Agronomist", height=320, margin=dict(l=0, r=0, t=10, b=0))
        hm.update_yaxes(autorange="reversed"); a.plotly_chart(hm, width="stretch")
        ag = go.Figure(go.Heatmap(z=(ai == mask).astype(int), colorscale=[[0, "#d32f2f"], [1, "#e8f5e9"]], showscale=False))
        ag.update_yaxes(autorange="reversed", scaleanchor="x", visible=False); ag.update_xaxes(visible=False)
        ag.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), title="Disagreement (red)")
        bb.plotly_chart(ag, width="stretch")

    with st.expander("💬 Messages with the farmer"):
        comment_thread("scan", sid, user)
    st.subheader("Submit validation")
    with st.form(f"sub_{sid}"):
        status = st.radio("Outcome", ["Validated", "Re-survey needed"], horizontal=True)
        notes = st.text_area("Observations / field notes", value=(prev or {}).get("notes") or "")
        rec = st.text_area("Recommendation for the farmer", value=(prev or {}).get("recommendation") or "")
        if st.form_submit_button("📨 Submit review", type="primary"):
            met = dict(accuracy=float(m["overall_accuracy"]), kappa=float(m["kappa"]), miou=float(m["miou"]),
                       bf1=float(bf["f1"]), cm=cm.tolist())
            db.add_review(sid, user["id"], "validated" if status == "Validated" else "resurvey", notes, rec, mask, met)
            st.toast("Review saved – the farmer can now see the validated boundaries ✅"); st.rerun()


def autotune(user):
    header("🎯 Auto-Tune K-Means")
    st.caption("Searches k, index set and smoothing for the configuration that best matches the agronomist's "
               "validated boundaries (maximises Cohen's κ).")
    done = [s for s in db.list_scans() if s["review_status"] in ("validated", "resurvey")]
    if not done:
        st.info("Validate at least one scan first."); return
    s = st.selectbox("Validated scan", done, format_func=_label)
    rev = db.get_review(s["id"]); cube, _ = db.load_scan_arrays(s["id"]); bands = bands_from_cube(cube)
    sets = {"NDVI": ["NDVI"], "NDVI+NDRE": ["NDVI", "NDRE"], "NDVI+GNDVI+NDRE+SAVI": ["NDVI", "GNDVI", "NDRE", "SAVI"],
            "NDVI+NDRE+NDWI+CIre": ["NDVI", "NDRE", "NDWI", "CIre"]}
    if st.button("🔍 Run search", type="primary"):
        rows, grid = [], list(itertools.product([2, 3, 4, 5], sets, [0, 3, 5]))
        bar = st.progress(0.0)
        for i, (k, nm, sm) in enumerate(grid):
            r = run_analysis(bands, sets[nm], (), k, sm)
            m = metrics(r["cls"], rev["mask"])
            rows.append(dict(k=k, features=nm, smoothing=sm, accuracy=round(m["overall_accuracy"], 3),
                             kappa=round(m["kappa"], 3), miou=round(m["miou"], 3),
                             boundary_f1=round(boundary_f1(r["cls"], rev["mask"], 3)["f1"], 3)))
            bar.progress((i + 1) / len(grid))
        st.session_state["tune"] = pd.DataFrame(rows).sort_values("kappa", ascending=False).reset_index(drop=True)
    df = st.session_state.get("tune")
    if df is not None:
        st.dataframe(df.head(15), width="stretch"); best = df.iloc[0]
        st.success(f"Best: k={int(best.k)}, {best.features}, smoothing={int(best.smoothing)} → κ={best.kappa}")
        if st.button("Apply best settings to this scan's AI zones"):
            r = run_analysis(bands, sets[best.features], (), int(best.k), int(best.smoothing))
            db.update_scan_zones(s["id"], r["cls"], r["params"], r["stats"]); st.toast("Scan re-zoned ✅"); st.rerun()


def all_fields(user):
    header("🗺️ All Fields")
    rows = []
    for f in db.list_fields():
        sc = db.list_scans(field_id=f["id"]); l = sc[0] if sc else None
        rows.append({"Field": f["name"], "Farmer": f["farmer_name"], "Crop": f["crop"], "Area (ha)": f["area_ha"],
                     "Scans": len(sc), "Last scan": l["created"][:10] if l else "–",
                     "Healthy %": round(l["stats"]["zone_pct"][0], 1) if l else None,
                     "Diseased %": round(l["stats"]["zone_pct"][2], 1) if l else None,
                     "Status": STATUS[l["review_status"]] if l else "–"})
    if not rows:
        st.info("No fields registered."); return
    df = pd.DataFrame(rows); st.dataframe(df, hide_index=True, width="stretch")
    d = df.dropna(subset=["Diseased %"]).sort_values("Diseased %", ascending=False)
    if len(d):
        fig = go.Figure(go.Bar(x=d["Field"], y=d["Diseased %"], marker_color=CLASS_COLORS[2]))
        fig.update_layout(title="Hotspots: share of field classed Diseased (%)", height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")


def analytics(user):
    header("📊 Model vs Agronomist Analytics")
    revs = db.list_reviews()
    if not revs:
        st.info("No validated scans yet."); return
    df = pd.DataFrame([{"review": r["id"], "date": r["created"][:10], **{k: r["metrics"].get(k) for k in ("accuracy", "kappa", "miou", "bf1")}} for r in revs])
    c = st.columns(4)
    c[0].metric("Reviews", len(df)); c[1].metric("Mean accuracy", f"{df.accuracy.mean() * 100:.1f}%")
    c[2].metric("Mean κ", f"{df.kappa.mean():.3f}"); c[3].metric("Mean boundary F1", f"{df.bf1.mean():.3f}")
    fig = go.Figure([go.Scatter(x=df.index + 1, y=df[k], name=k, mode="lines+markers") for k in ("accuracy", "kappa", "miou", "bf1")])
    fig.update_layout(title="Agreement per review", xaxis_title="Review #", height=320, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")
    cm = sum(np.array(r["metrics"]["cm"]) for r in revs if "cm" in r["metrics"])
    cmn = cm / cm.sum(1, keepdims=True).clip(1)
    hm = go.Figure(go.Heatmap(z=cmn, x=CLASS_NAMES, y=CLASS_NAMES, colorscale="Blues", zmin=0, zmax=1,
                              text=[[f"{cmn[i, j]:.0%}" for j in range(3)] for i in range(3)], texttemplate="%{text}"))
    hm.update_layout(title="Aggregate confusion matrix (row-normalised)", xaxis_title="K-Means", yaxis_title="Agronomist",
                     height=380, margin=dict(l=0, r=0, t=40, b=0)); hm.update_yaxes(autorange="reversed")
    st.plotly_chart(hm, width="stretch")


def photo_reviews(user):
    from ui.photo import render_reviews
    render_reviews(user)


def lab(user):
    from ui.lab import render_lab
    render_lab()


ROUTES = dict(zip(PAGES, [dashboard, validate, photo_reviews, autotune, all_fields, analytics, lab]))
