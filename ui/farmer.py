"""Farmer-facing pages."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from czs import CLASS_COLORS, CLASS_NAMES, db
from czs.advice import DISCLAIMER, advise
from czs.config import ALERT_DISEASED_PCT
from czs.indices import INDEX_INFO, compute_indices
from czs.pipeline import DEFAULT_FEATURES, bands_from_cube, cube_from_bands, run_analysis, zone_stats
from czs.report import build_report
from czs.validation import boundary_map
from ui.components import comment_thread, gauge, header
from ui.photo import render_photo
from ui.common import (STATUS, bands_input, class_heatmap, legend, png_bytes, zone_metrics, zone_table)

CROPS = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Potato", "Soybean", "Vegetables", "Other"]
PAGES = ["🏠 Overview", "🌱 My Fields", "📷 Photo Analysis", "🛰️ Multispectral Scan", "📈 History & Reports", "🔀 Compare Scans", "📒 Field Journal"]


def _scan_label(s):
    return f"#{s['id']} · {s['created'][:16].replace('T', ' ')} · {STATUS[s['review_status']]}"


def official_zones(scan):
    """Agronomist-validated mask if available, else the K-Means zones."""
    cube, cls = db.load_scan_arrays(scan["id"])
    rev = db.get_review(scan["id"])
    mask = rev["mask"] if rev and rev["mask"] is not None else None
    return cube, cls, mask, rev


# ------------------------------------------------------------------ pages
def overview(user):
    header(f"Welcome, {user['name']} 👋")
    fields, scans = db.list_fields(user["id"]), db.list_scans(owner_id=user["id"])
    photos = db.list_photos(owner_id=user["id"])
    c = st.columns(5)
    c[0].metric("Fields", len(fields)); c[1].metric("Multispectral scans", len(scans)); c[2].metric("Photos analysed", len(photos))
    c[3].metric("Awaiting review", sum(s["review_status"] == "pending" for s in scans) + sum(p["review_status"] == "pending" for p in photos))
    c[4].metric("Reviewed / validated", sum(s["review_status"] == "validated" for s in scans) + sum(p["review_status"] == "reviewed" for p in photos))
    if not fields:
        st.info("Start by adding a field under **🌱 My Fields**, then upload a crop photo in **📷 Photo Analysis**.")
        return
    st.subheader("Latest status per field")
    for f in fields:
        fs = [s for s in scans if s["field_id"] == f["id"]]
        fp = [p for p in photos if p["field_id"] == f["id"]]
        with st.container(border=True):
            a, b = st.columns([1, 2])
            a.markdown(f"**{f['name']}**  \n{f['crop']} · {f['area_ha'] or '–'} ha  \n{f['location'] or ''}")
            if fp:
                b.caption(f"📷 Latest photo score **{fp[0]['details']['health_score']:.0f}/100** · {fp[0]['details']['grade']} · {fp[0]['created'][:10]}")
            if not fs:
                if not fp:
                    b.caption("No scans or photos yet.")
                continue
            s = fs[0]; zp = s["stats"]["zone_pct"]
            b.caption(f"Latest scan {s['created'][:10]} · {STATUS[s['review_status']]}")
            if zp[2] >= ALERT_DISEASED_PCT:
                b.error(f"⚠️ {zp[2]:.0f}% of this field is classed Diseased – see recommendations.")
            fig = go.Figure(go.Bar(x=zp[::-1], y=CLASS_NAMES[::-1], orientation="h",
                                   marker_color=CLASS_COLORS[::-1], text=[f"{v:.0f}%" for v in zp[::-1]]))
            fig.update_layout(height=130, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False))
            b.plotly_chart(fig, width="stretch", key=f"ov_{f['id']}")


def my_fields(user):
    header("🌱 My Fields")
    with st.form("add_field", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Field name")
        crop = c2.selectbox("Crop", CROPS)
        area = c1.number_input("Area (hectares)", 0.1, 5000.0, 5.0, 0.5)
        loc = c2.text_input("Location / village (optional)")
        if st.form_submit_button("➕ Add field", type="primary"):
            if len(name.strip()) < 2:
                st.error("Enter a field name.")
            else:
                db.add_field(user["id"], name.strip(), crop, area, loc.strip()); st.toast("Field added ✅"); st.rerun()
    fields = db.list_fields(user["id"])
    if fields:
        st.dataframe(pd.DataFrame(fields)[["name", "crop", "area_ha", "location", "created"]], hide_index=True,
                     width="stretch")
        with st.expander("🗑️ Delete a field (and all its scans)"):
            fsel = st.selectbox("Field", fields, format_func=lambda f: f["name"], key="del_f")
            if st.checkbox("I understand this cannot be undone") and st.button("Delete field"):
                db.delete_field(fsel["id"], user["id"]); st.rerun()


def new_scan(user):
    header("🛰️ Multispectral Scan")
    fields = db.list_fields(user["id"])
    if not fields:
        st.info("Add a field first under **🌱 My Fields**."); return
    fld = st.selectbox("Field", fields, format_func=lambda f: f"{f['name']} ({f['crop']})")
    bands, src = bands_input("ns")
    with st.expander("⚙️ Advanced settings"):
        feats = st.multiselect("Vegetation indices", list(INDEX_INFO), DEFAULT_FEATURES)
        c1, c2 = st.columns(2)
        k = c1.slider("Zones (k)", 2, 6, 3); smooth = c2.select_slider("Smoothing", [0, 3, 5, 7], 3)
    if bands is not None and st.button("▶️ Run analysis", type="primary") and feats:
        with st.spinner("Computing indices and clustering…"):
            r = run_analysis(bands, feats, (), k, smooth)
        st.session_state["draft"] = dict(field_id=fld["id"], area=fld["area_ha"], cube=cube_from_bands(bands),
                                         res=r, source=src)
    d = st.session_state.get("draft")
    if not d:
        return
    st.divider(); st.subheader("Result preview")
    r = d["res"]; legend(); zone_metrics(r["cls"], d["area"])
    t1, t2 = st.tabs(["Zone map", "NDVI map"])
    t1.plotly_chart(class_heatmap(r["cls"], "Stress zones", boundary_map(r["cls"])), width="stretch")
    f2 = go.Figure(go.Heatmap(z=r["indices"]["NDVI"], colorscale="RdYlGn"))
    f2.update_yaxes(autorange="reversed", scaleanchor="x", visible=False); f2.update_xaxes(visible=False)
    f2.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0)); t2.plotly_chart(f2, width="stretch")
    req = st.checkbox("Send to an agronomist for validation", True)
    c1, c2 = st.columns(2)
    if c1.button("💾 Save scan", type="primary"):
        sid = db.save_scan(d["field_id"], d["cube"], r["cls"], r["params"], r["stats"], d["source"], req)
        st.session_state.pop("draft"); st.toast(f"Scan #{sid} saved ✅"); st.rerun()
    if c2.button("Discard"):
        st.session_state.pop("draft"); st.rerun()


def history(user):
    header("📈 History & Reports")
    fields = db.list_fields(user["id"])
    if not fields:
        st.info("No fields yet."); return
    fld = st.selectbox("Field", fields, format_func=lambda f: f["name"])
    scans = db.list_scans(field_id=fld["id"])
    if not scans:
        st.info("No scans for this field yet."); return
    if len(scans) > 1:
        df = pd.DataFrame([{"date": s["created"][:10], **{CLASS_NAMES[i]: s["stats"]["zone_pct"][i] for i in range(3)}}
                           for s in reversed(scans)])
        fig = go.Figure([go.Scatter(x=df["date"], y=df[n], name=n, mode="lines+markers", stackgroup="one",
                                    line=dict(color=c)) for n, c in zip(CLASS_NAMES, CLASS_COLORS)])
        for j in db.list_journal(fld["id"]):
            fig.add_shape(type="line", x0=j["day"], x1=j["day"], y0=0, y1=100, line=dict(color="#0277bd", width=2, dash="dot"))
            fig.add_annotation(x=j["day"], y=100, text=j["action"], showarrow=False, textangle=-90, yanchor="top", font=dict(size=10, color="#0277bd"))
        fig.update_layout(title="Zone share over time (%)  ·  blue lines = journal actions", height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    s = st.selectbox("Scan", scans, format_func=_scan_label)
    cube, ai, mask, rev = official_zones(s)
    official = mask if mask is not None else ai
    indices = compute_indices(bands_from_cube(cube))
    stats = zone_stats(official, indices)
    st.caption(f"{STATUS[s['review_status']]} · source: {s['source']}")
    legend(); zone_metrics(official, s["area_ha"])
    t1, t2, t3, t4 = st.tabs(["🗺️ Zones", "💡 Recommendations", "💬 Messages", "📄 Report"])
    with t1:
        if mask is not None:
            a, b = st.columns(2)
            a.plotly_chart(class_heatmap(mask, "Agronomist-validated zones", boundary_map(mask)), width="stretch")
            b.plotly_chart(class_heatmap(ai, "K-Means (automatic)", boundary_map(ai)), width="stretch")
        else:
            st.plotly_chart(class_heatmap(ai, "K-Means zones (not yet validated)", boundary_map(ai)), width="stretch")
        st.dataframe(zone_table(official, s["area_ha"]), hide_index=True, width="stretch")
        st.download_button("⬇️ Zone map (PNG)", png_bytes(official), f"scan_{s['id']}_zones.png", "image/png")
    with t2:
        if rev:
            st.success(f"**{rev['agronomist_name']}** · {rev['created'][:10]}")
            st.markdown(f"**Notes:** {rev['notes'] or '–'}\n\n**Recommendation:** {rev['recommendation'] or '–'}")
        elif s["review_status"] == "pending":
            st.info("Waiting for an agronomist to validate this scan.")
        for prio, title, text in advise(stats, s["area_ha"]):
            {"High": st.error, "Medium": st.warning, "Low": st.success}[prio](f"**{title}**  \n{text}")
        st.caption(DISCLAIMER)
    with t3:
        comment_thread("scan", s["id"], user)
    with t4:
        html = build_report(s, official, ai, rev, advise(stats, s["area_ha"]), s["area_ha"], DISCLAIMER)
        st.download_button("⬇️ Download report (HTML – print to PDF)", html, f"report_scan_{s['id']}.html", "text/html")
        if s["review_status"] == "not_requested" and st.button("📨 Send to agronomist for validation"):
            with db.conn() as c:
                c.execute("UPDATE scans SET review_status='pending' WHERE id=?", (s["id"],))
            st.rerun()
        if st.button("🗑️ Delete this scan"):
            db.delete_scan(s["id"]); st.rerun()


def compare(user):
    header("🔀 Compare Scans")
    fields = db.list_fields(user["id"])
    if not fields:
        st.info("No fields yet."); return
    fld = st.selectbox("Field", fields, format_func=lambda f: f["name"])
    scans = [s for s in db.list_scans(field_id=fld["id"])]
    if len(scans) < 2:
        st.info("You need at least two scans of the same field."); return
    c1, c2 = st.columns(2)
    a = c1.selectbox("Earlier scan", scans, index=len(scans) - 1, format_func=_scan_label)
    b = c2.selectbox("Later scan", scans, index=0, format_func=_scan_label)
    za, zb = official_zones(a), official_zones(b)
    ca = za[2] if za[2] is not None else za[1]; cb = zb[2] if zb[2] is not None else zb[1]
    if ca.shape != cb.shape:
        st.error("Scans have different resolutions and cannot be compared."); return
    delta = cb.astype(int) - ca.astype(int)  # + = worse
    cat = np.sign(delta) + 1  # 0 improved, 1 same, 2 worse
    fig = go.Figure(go.Heatmap(z=cat, zmin=0, zmax=2, showscale=False,
                               colorscale=[[0, "#2e7d32"], [1 / 3, "#2e7d32"], [1 / 3, "#bdbdbd"], [2 / 3, "#bdbdbd"],
                                           [2 / 3, "#c62828"], [1, "#c62828"]]))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", visible=False); fig.update_xaxes(visible=False)
    fig.update_layout(title="Change map (green = improved · grey = unchanged · red = worsened)", height=450,
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")
    m = st.columns(3)
    m[0].metric("Improved", f"{(delta < 0).mean() * 100:.1f}%"); m[1].metric("Unchanged", f"{(delta == 0).mean() * 100:.1f}%")
    m[2].metric("Worsened", f"{(delta > 0).mean() * 100:.1f}%")




JOURNAL_ACTIONS = ["Irrigation", "Fertiliser", "Fungicide / pesticide", "Herbicide", "Scouting", "Sowing / planting", "Pruning", "Harvest", "Other"]


def journal(user):
    import datetime
    header("📒 Field Journal")
    fields = db.list_fields(user["id"])
    if not fields:
        st.info("Add a field first."); return
    fld = st.selectbox("Field", fields, format_func=lambda f: f["name"], key="jr_field")
    with st.form("jr_add", clear_on_submit=True):
        c1, c2 = st.columns(2)
        day = c1.date_input("Date", datetime.date.today()); act = c2.selectbox("Action", JOURNAL_ACTIONS)
        notes = st.text_input("Details (product, dose, area…)")
        if st.form_submit_button("➕ Add entry", type="primary"):
            db.add_journal(fld["id"], user["id"], str(day), act, notes.strip()); st.toast("Entry added ✅"); st.rerun()
    rows = db.list_journal(fld["id"])
    if not rows:
        st.caption("No entries yet."); return
    st.dataframe(pd.DataFrame(rows)[["day", "action", "notes"]].rename(columns=str.title), hide_index=True, width="stretch")
    e = st.selectbox("Remove an entry", rows, format_func=lambda r: f"{r['day']} · {r['action']} · {r['notes'] or ''}")
    if st.button("🗑️ Delete entry"):
        db.delete_journal(e["id"], [fld["id"]]); st.rerun()
    st.caption("Entries appear as blue lines on the trend chart in **📈 History & Reports**, so you can see how the crop responded.")


ROUTES = dict(zip(PAGES, [overview, my_fields, lambda u: render_photo(u), new_scan, history, compare, journal]))
