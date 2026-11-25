"""Crop Stress Zoning – multi-user web app.  Run:  streamlit run app.py"""
import streamlit as st

if tuple(int(x) for x in st.__version__.split(".")[:2]) < (1, 50):
    raise RuntimeError(f"Streamlit {st.__version__} is too old - this app needs >=1.50 "
                       "(run: pip install -U -r requirements.txt).")

from czs import db
from czs.seed import seed_if_empty

st.set_page_config(page_title="Crop Stress Zoning", page_icon="🌾", layout="wide")


@st.cache_resource
def _bootstrap():
    db.init_db()
    seed_if_empty()
    return True


_bootstrap()

from ui import theme  # noqa: E402
theme.apply()

from ui import agronomist, farmer  # noqa: E402
from ui.auth_ui import render_login  # noqa: E402

user = st.session_state.get("user")
if not user:
    render_login()
    st.stop()

routes = farmer.ROUTES if user["role"] == "farmer" else agronomist.ROUTES
with st.sidebar:
    st.markdown("### 🌾 Crop Stress Zoning")
    st.markdown(f"**{user['name']}**  \n<span class='role-pill'>{'👨‍🌾 Farmer' if user['role'] == 'farmer' else '🔬 Agronomist'}</span>",
                unsafe_allow_html=True)
    page = st.radio("Navigate", list(routes), label_visibility="collapsed", key="nav")
    if st.button("Log out", width="stretch"):
        st.session_state.clear()
        st.rerun()
    st.divider()

routes[page](user)
