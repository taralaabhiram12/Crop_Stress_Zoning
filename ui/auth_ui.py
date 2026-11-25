"""Login / registration screen."""
import streamlit as st

from czs import auth
from czs.config import seed_demo
from ui.theme import login_hero


def render_login():
    login_hero()
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        t_login, t_reg = st.tabs(["🔐 Log in", "📝 Create account"])
        with t_login:
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", type="primary", width="stretch"):
                    user, msg = auth.authenticate(u, p)
                    if user:
                        st.session_state["user"] = user
                        st.rerun()
                    st.error(msg)
            if seed_demo():
                st.caption("Demo accounts → farmer: `demo_farmer` / `Farmer@123` · "
                           "agronomist: `demo_agronomist` / `Agro@1234`")
        with t_reg:
            with st.form("register"):
                name = st.text_input("Full name")
                un = st.text_input("Choose a username")
                pw = st.text_input("Password", type="password", help="8+ characters with a letter and a digit")
                pw2 = st.text_input("Confirm password", type="password")
                role = st.radio("I am a…", ["Farmer", "Agronomist"], horizontal=True)
                code = st.text_input("Agronomist invitation code", type="password",
                                     help="Required for agronomist accounts. Ask your administrator.")
                if st.form_submit_button("Create account", width="stretch"):
                    if pw != pw2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = auth.register(un, name, pw, role.lower(), code)
                        (st.success if ok else st.error)(msg)
