import os, sys, tempfile
os.environ["DATA_DIR"] = tempfile.mkdtemp()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pytest
from czs import auth, db


def test_auth_rules():
    db.init_db()
    assert not auth.register("ab", "X Y", "pass1234", "farmer")[0]
    assert not auth.register("user_ok", "X Y", "short", "farmer")[0]
    assert not auth.register("agro1", "Dr X", "pass1234", "agronomist", "wrong")[0]
    assert auth.register("farmer_t", "Test Farmer", "pass1234", "farmer")[0]
    assert not auth.register("farmer_t", "Dup", "pass1234", "farmer")[0]
    assert auth.authenticate("farmer_t", "pass1234")[0]["role"] == "farmer"
    assert auth.authenticate("farmer_t", "nope")[0] is None


def test_lockout():
    auth.register("locky", "Lock User", "pass1234", "farmer")
    for _ in range(auth.MAX_FAILS):
        auth.authenticate("locky", "bad")
    assert auth.authenticate("locky", "pass1234")[0] is None  # locked even with right password


def test_scan_and_review_roundtrip():
    db.init_db()
    uid = db.get_user("farmer_t")["id"]
    fid = db.add_field(uid, "F1", "Wheat", 5, "x")
    cube, cls = np.random.rand(16, 16, 5), np.random.randint(0, 3, (16, 16))
    sid = db.save_scan(fid, cube, cls, {"k": 3}, {"zone_pct": [1, 1, 1]}, "test")
    assert db.list_scans(owner_id=uid)[0]["review_status"] == "pending"
    _, c2 = db.load_scan_arrays(sid); assert (c2 == cls).all()
    aid = auth.register("agro_t", "Dr T", "pass1234", "agronomist", auth.agronomist_code()) and db.get_user("agro_t")["id"]
    db.add_review(sid, aid, "validated", "ok", "spray", cls, {"kappa": 1.0})
    assert db.get_scan(sid)["review_status"] == "validated" and db.get_review(sid)["mask"].shape == (16, 16)
    assert db.delete_field(fid, uid) and not db.list_scans(field_id=fid)


def test_ownership_enforced():
    other = db.get_user("agro_t")["id"]
    fid = db.add_field(db.get_user("farmer_t")["id"], "Mine", "Rice", 1, "")
    assert not db.delete_field(fid, other)


def test_ui_end_to_end():
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    st.cache_resource.clear()
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()
    assert not at.exception
    auth.register("ui_farmer", "UI Farmer", "pass1234", "farmer")
    at.text_input[0].set_value("ui_farmer"); at.text_input[1].set_value("pass1234"); at.button[0].click(); at.run()
    assert at.session_state["user"]["role"] == "farmer" and not at.exception


def test_photo_flow_and_review():
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    from czs import db
    st.cache_resource.clear()
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240).run()
    at.text_input[0].set_value("ui_farmer"); at.text_input[1].set_value("pass1234"); at.button[0].click(); at.run()
    uid = db.get_user("ui_farmer")["id"]; db.add_field(uid, "Photo Field", "Wheat", 3, "")
    at.sidebar.radio[0].set_value("📷 Photo Analysis").run()
    [b for b in at.button if "sample field" in b.label][0].click(); at.run()
    assert not at.exception and len(at.metric) >= 6
    [b for b in at.button if "Save photo analysis" in b.label][0].click(); at.run()
    assert not at.exception
    photos = db.list_photos(owner_id=uid); assert photos and photos[0]["review_status"] == "pending"
    db.review_photo(photos[0]["id"], db.get_user("agro_t")["id"], "Check for leaf spot")
    db.add_comment("photo", photos[0]["id"], uid, "Thanks!")
    at.sidebar.radio[0].set_value("📷 Photo Analysis").run()
    assert not at.exception
    assert db.list_photos(owner_id=uid)[0]["review_status"] == "reviewed" and len(db.list_comments("photo", photos[0]["id"])) == 1
