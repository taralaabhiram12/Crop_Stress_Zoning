"""Reusable UI building blocks."""
import html

import plotly.graph_objects as go
import streamlit as st

from czs import db

SUBS = {
    "🌱 My Fields": "Register your fields and keep their details in one place.",
    "🛰️ Multispectral Scan": "Full K-Means zoning from multispectral drone or satellite imagery.",
    "📈 History & Reports": "Track field health over time, read agronomist feedback and download reports.",
    "🔀 Compare Scans": "See exactly where the crop improved or got worse between two dates.",
    "📒 Field Journal": "Log irrigation, spraying and fertiliser so you can link actions to crop response.",
    "✅ Validate Scan": "Correct the AI zone boundaries and record your expert validation.",
    "🎯 Auto-Tune K-Means": "Find the K-Means settings that best match your validated boundaries.",
    "🗺️ All Fields": "Monitor every registered field and find the hotspots.",
    "📊 Model vs Agronomist Analytics": "How closely do the automatic zones agree with expert markings?",
    "🔬 Analysis Lab": "Full-control exploration of indices, clustering and validation.",
    "📷 Photo Analysis": "Upload a normal phone or drone photo and get a crop-stress assessment in seconds.",
    "📷 Photo Reviews": "Diagnose farmer-submitted crop photos and send your advice back.",
}


def header(title, sub=None):
    sub = sub if sub is not None else SUBS.get(title, "")
    st.markdown(f"<div class='ph'><div class='ph-t'>{html.escape(title)}</div>"
                f"<div class='ph-s'>{html.escape(sub)}</div></div>", unsafe_allow_html=True)


def gauge(score, title="Crop health score", height=250):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title=dict(text=title, font=dict(size=15)),
        number=dict(suffix="/100", font=dict(size=38)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#1b5e20", thickness=0.28),
                   steps=[dict(range=[0, 30], color="#ef9a9a"), dict(range=[30, 50], color="#ffcc80"),
                          dict(range=[50, 70], color="#fff59d"), dict(range=[70, 85], color="#c5e1a5"),
                          dict(range=[85, 100], color="#81c784")])))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=0))
    return fig


def grade_badge(grade):
    col = {"Excellent": "#2e7d32", "Good": "#558b2f", "Needs attention": "#f9a825", "Poor": "#ef6c00",
           "Critical": "#c62828"}.get(grade, "#607d8b")
    st.markdown(f"<div style='text-align:center'><span style='background:{col};color:#fff;padding:5px 18px;"
                f"border-radius:999px;font-weight:700'>{html.escape(grade)}</span></div>", unsafe_allow_html=True)


def comment_thread(kind, ref_id, user):
    for c in db.list_comments(kind, ref_id):
        agr = c["role"] == "agronomist"
        with st.chat_message("assistant" if agr else "user", avatar="🔬" if agr else "👨‍🌾"):
            st.markdown(f"**{c['name']}** · {c['created'][:16].replace('T', ' ')}")
            st.write(c["text"])
    with st.form(f"cm_{kind}_{ref_id}", clear_on_submit=True):
        t = st.text_area("Message", height=80, label_visibility="collapsed",
                         placeholder="Write a message to the farmer / agronomist…")
        if st.form_submit_button("💬 Send") and t.strip():
            db.add_comment(kind, ref_id, user["id"], t); st.rerun()
