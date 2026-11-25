"""Visual theme: CSS, Plotly template and login artwork."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

GREEN, DARK, LIGHT = "#2e7d32", "#1b5e20", "#e8f1e4"

pio.templates["crop"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, sans-serif", color="#1b2e1b"),
    colorway=["#2e7d32", "#f9a825", "#c62828", "#0277bd", "#6a1b9a", "#00838f"],
    title=dict(font=dict(size=16)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"))
pio.templates.default = "plotly+crop"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.stApp {{ background: linear-gradient(180deg, #f6faf4 0%, #eef5ea 100%); }}
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}

/* page titles */
h1 {{ color: {DARK}; font-weight: 700; letter-spacing: -0.5px;
      border-bottom: 4px solid #a5d6a7; padding-bottom: .35rem; display: inline-block; }}
h2, h3 {{ color: #1b3a1d; font-weight: 600; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #e3f0de 0%, #d3e8cc 100%);
      border-right: 1px solid #b7d7b0; }}
section[data-testid="stSidebar"] [role="radiogroup"] label {{ background: rgba(255,255,255,.55);
      border-radius: 10px; padding: 6px 10px; margin-bottom: 4px; transition: all .15s; width: 100%; }}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: #fff; transform: translateX(3px); }}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
      background: {GREEN}; color: #fff; box-shadow: 0 2px 8px rgba(46,125,50,.35); }}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {{ color: #fff !important; }}
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}

/* metric cards */
div[data-testid="stMetric"] {{ background: #fff; border: 1px solid #dcebd8; border-left: 6px solid {GREEN};
      border-radius: 14px; padding: 14px 18px; box-shadow: 0 2px 10px rgba(30,80,30,.08); }}
div[data-testid="stMetricLabel"] {{ color: #4a6b4c; font-weight: 500; }}
div[data-testid="stMetricValue"] {{ color: {DARK}; font-weight: 700; }}

/* containers / cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-radius: 16px;
      box-shadow: 0 2px 12px rgba(30,80,30,.07); }}
div[data-testid="stPlotlyChart"] {{ background: #fff; border-radius: 14px; padding: 6px;
      box-shadow: 0 2px 10px rgba(30,80,30,.07); }}
div[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

/* buttons */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {{
      border-radius: 999px; font-weight: 600; border: 1px solid #a5d6a7; transition: all .15s; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {GREEN}; transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(46,125,50,.25); }}
button[kind="primary"] {{ background: linear-gradient(135deg, #43a047, {DARK}); border: none; color: #fff; }}

/* tabs */
button[data-baseweb="tab"] {{ font-weight: 600; border-radius: 10px 10px 0 0; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {DARK}; background: #fff; }}

/* alerts */
div[data-testid="stAlert"] {{ border-radius: 12px; }}

/* login hero */
.hero {{ background: linear-gradient(135deg, #1b5e20 0%, #43a047 60%, #9ccc65 100%); color: #fff;
      border-radius: 22px; padding: 26px 30px 12px; box-shadow: 0 10px 30px rgba(27,94,32,.35); margin-bottom: 18px; }}
.hero h1 {{ color: #fff; border: none; margin: 0; font-size: 2.2rem; }}
.hero p {{ color: #e8f5e9; margin: 4px 0 12px; }}
.chips span {{ display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
      border-radius: 999px; padding: 3px 12px; margin: 0 6px 8px 0; font-size: .82rem; }}
.role-pill {{ display:inline-block; background:{GREEN}; color:#fff; border-radius:999px; padding:2px 12px;
      font-size:.8rem; font-weight:600; }}

/* page header banner */
.ph {{ background: linear-gradient(120deg, #1b5e20 0%, #43a047 65%, #9ccc65 100%); color: #fff; border-radius: 18px;
      padding: 20px 28px; margin-bottom: 20px; box-shadow: 0 8px 22px rgba(27,94,32,.28); }}
.ph-t {{ font-size: 1.8rem; font-weight: 700; letter-spacing: -.3px; }}
.ph-s {{ opacity: .92; margin-top: 2px; }}
div[data-testid="stToolbar"], .stDeployButton {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stChatMessage"] {{ background: #fff; border-radius: 14px; box-shadow: 0 1px 6px rgba(30,80,30,.08); }}
</style>
"""

FIELD_SVG = """
<svg viewBox="0 0 600 150" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:14px;margin-top:6px">
 <defs><filter id="b"><feGaussianBlur stdDeviation="9"/></filter>
  <pattern id="rows" width="14" height="150" patternUnits="userSpaceOnUse">
   <rect width="14" height="150" fill="#2e7d32"/><rect width="5" height="150" fill="#1b5e20" opacity=".45"/></pattern></defs>
 <rect width="600" height="150" fill="url(#rows)"/>
 <g filter="url(#b)" opacity=".85">
  <ellipse cx="170" cy="85" rx="75" ry="42" fill="#f9a825"/><ellipse cx="190" cy="95" rx="38" ry="22" fill="#c62828"/>
  <ellipse cx="440" cy="60" rx="85" ry="38" fill="#f9a825"/><ellipse cx="455" cy="55" rx="40" ry="20" fill="#c62828"/></g>
 <g fill="none" stroke="#fff" stroke-width="2.5" stroke-dasharray="7 5" opacity=".9">
  <ellipse cx="170" cy="85" rx="75" ry="42"/><ellipse cx="440" cy="60" rx="85" ry="38"/></g>
 <text x="14" y="138" fill="#fff" font-size="13" font-family="Arial" font-weight="bold">K-Means stress zones · agronomist-validated boundaries</text>
</svg>"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def login_hero():
    st.markdown(f"""<div class="hero"><h1>🌾 Crop Stress Zoning</h1>
<p>Multispectral crop-health mapping with agronomist-validated boundaries</p>
<div class="chips"><span>🛰️ Vegetation indices</span><span>🧩 K-Means zones</span>
<span>✅ Agronomist validation</span><span>📈 Field trends</span></div>{FIELD_SVG}</div>""", unsafe_allow_html=True)
