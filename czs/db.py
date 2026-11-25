"""SQLite persistence for users, fields, scans and agronomist reviews."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

from .config import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('farmer','agronomist')), pw_hash TEXT NOT NULL, salt TEXT NOT NULL,
  fail_count INTEGER DEFAULT 0, locked_until REAL DEFAULT 0, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fields(
  id INTEGER PRIMARY KEY AUTOINCREMENT, owner_id INTEGER NOT NULL REFERENCES users(id),
  name TEXT NOT NULL, crop TEXT, area_ha REAL, location TEXT, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scans(
  id INTEGER PRIMARY KEY AUTOINCREMENT, field_id INTEGER NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
  created TEXT NOT NULL, source TEXT, params TEXT, stats TEXT,
  review_status TEXT NOT NULL DEFAULT 'not_requested');
CREATE TABLE IF NOT EXISTS reviews(
  id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
  agronomist_id INTEGER NOT NULL REFERENCES users(id), created TEXT NOT NULL, status TEXT NOT NULL,
  notes TEXT, recommendation TEXT, metrics TEXT);
CREATE TABLE IF NOT EXISTS photos(
  id INTEGER PRIMARY KEY AUTOINCREMENT, field_id INTEGER NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
  created TEXT NOT NULL, note TEXT, details TEXT, review_status TEXT NOT NULL DEFAULT 'not_requested',
  diagnosis TEXT, reviewer_id INTEGER, reviewed TEXT);
CREATE TABLE IF NOT EXISTS comments(
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, ref_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL REFERENCES users(id), created TEXT NOT NULL, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS journal(
  id INTEGER PRIMARY KEY AUTOINCREMENT, field_id INTEGER NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL, day TEXT NOT NULL, action TEXT NOT NULL, notes TEXT, created TEXT NOT NULL);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn():
    d = data_dir(); d.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(d / "app.db", timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    for sub in ("scans", "reviews", "photos"):
        (data_dir() / sub).mkdir(parents=True, exist_ok=True)
    with conn() as c:
        c.executescript(SCHEMA)


def _d(row):
    return dict(row) if row is not None else None


# ------------------------------------------------------------------ users
def create_user(username, name, role, pw_hash, salt):
    try:
        with conn() as c:
            cur = c.execute("INSERT INTO users(username,name,role,pw_hash,salt,created) VALUES(?,?,?,?,?,?)",
                            (username, name, role, pw_hash, salt, now()))
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("Username already taken.")


def get_user(username):
    with conn() as c:
        return _d(c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone())


def count_users():
    with conn() as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def set_login_state(user_id, fail_count, locked_until=0.0):
    with conn() as c:
        c.execute("UPDATE users SET fail_count=?, locked_until=? WHERE id=?", (fail_count, locked_until, user_id))


# ------------------------------------------------------------------ fields
def add_field(owner_id, name, crop, area_ha, location):
    with conn() as c:
        return c.execute("INSERT INTO fields(owner_id,name,crop,area_ha,location,created) VALUES(?,?,?,?,?,?)",
                         (owner_id, name, crop, area_ha, location, now())).lastrowid


def list_fields(owner_id=None):
    q = ("SELECT f.*, u.name AS farmer_name FROM fields f JOIN users u ON u.id=f.owner_id")
    args = ()
    if owner_id is not None:
        q += " WHERE f.owner_id=?"; args = (owner_id,)
    with conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY f.name", args)]


def delete_field(field_id, owner_id):
    with conn() as c:
        ok = c.execute("SELECT 1 FROM fields WHERE id=? AND owner_id=?", (field_id, owner_id)).fetchone()
    if not ok:
        return False
    for s in list_scans(field_id=field_id):
        delete_scan(s["id"])
    for p in list_photos(field_id=field_id):
        delete_photo(p["id"])
    with conn() as c:
        c.execute("DELETE FROM fields WHERE id=?", (field_id,))
    return True


# ------------------------------------------------------------------ scans
def _scan(r):
    d = dict(r)
    d["params"] = json.loads(d["params"] or "{}"); d["stats"] = json.loads(d["stats"] or "{}")
    return d


def save_scan(field_id, cube, cls, params, stats, source, request_review=True):
    with conn() as c:
        sid = c.execute("INSERT INTO scans(field_id,created,source,params,stats,review_status) VALUES(?,?,?,?,?,?)",
                        (field_id, now(), source, json.dumps(params), json.dumps(stats),
                         "pending" if request_review else "not_requested")).lastrowid
    np.savez_compressed(data_dir() / "scans" / f"{sid}.npz", cube=cube.astype(np.float16), cls=cls.astype(np.uint8))
    return sid


def update_scan_zones(scan_id, cls, params, stats):
    p = data_dir() / "scans" / f"{scan_id}.npz"
    z = np.load(p)
    np.savez_compressed(p, cube=z["cube"], cls=cls.astype(np.uint8))
    with conn() as c:
        c.execute("UPDATE scans SET params=?, stats=? WHERE id=?", (json.dumps(params), json.dumps(stats), scan_id))


_SCAN_Q = ("SELECT s.*, f.name AS field_name, f.crop, f.area_ha, f.owner_id, u.name AS farmer_name "
           "FROM scans s JOIN fields f ON f.id=s.field_id JOIN users u ON u.id=f.owner_id ")


def list_scans(field_id=None, owner_id=None, status=None):
    where, args = [], []
    if field_id is not None: where.append("s.field_id=?"); args.append(field_id)
    if owner_id is not None: where.append("f.owner_id=?"); args.append(owner_id)
    if status is not None: where.append("s.review_status=?"); args.append(status)
    q = _SCAN_Q + (("WHERE " + " AND ".join(where)) if where else "") + " ORDER BY s.created DESC, s.id DESC"
    with conn() as c:
        return [_scan(r) for r in c.execute(q, args)]


def get_scan(scan_id):
    with conn() as c:
        r = c.execute(_SCAN_Q + "WHERE s.id=?", (scan_id,)).fetchone()
    return _scan(r) if r else None


def load_scan_arrays(scan_id):
    z = np.load(data_dir() / "scans" / f"{scan_id}.npz")
    return z["cube"].astype(np.float32), z["cls"].astype(np.uint8)


def delete_scan(scan_id):
    with conn() as c:
        rids = [r[0] for r in c.execute("SELECT id FROM reviews WHERE scan_id=?", (scan_id,))]
        c.execute("DELETE FROM reviews WHERE scan_id=?", (scan_id,))
        c.execute("DELETE FROM scans WHERE id=?", (scan_id,))
    for rid in rids:
        (data_dir() / "reviews" / f"{rid}.npy").unlink(missing_ok=True)
    (data_dir() / "scans" / f"{scan_id}.npz").unlink(missing_ok=True)


# ------------------------------------------------------------------ reviews
def add_review(scan_id, agronomist_id, status, notes, recommendation, mask, metrics):
    with conn() as c:
        rid = c.execute("INSERT INTO reviews(scan_id,agronomist_id,created,status,notes,recommendation,metrics) "
                        "VALUES(?,?,?,?,?,?,?)", (scan_id, agronomist_id, now(), status, notes,
                                                  recommendation, json.dumps(metrics))).lastrowid
        c.execute("UPDATE scans SET review_status=? WHERE id=?", (status, scan_id))
    np.save(data_dir() / "reviews" / f"{rid}.npy", mask.astype(np.uint8))
    return rid


def get_review(scan_id):
    with conn() as c:
        r = c.execute("SELECT r.*, u.name AS agronomist_name FROM reviews r JOIN users u ON u.id=r.agronomist_id "
                      "WHERE r.scan_id=? ORDER BY r.id DESC LIMIT 1", (scan_id,)).fetchone()
    if not r:
        return None
    d = dict(r); d["metrics"] = json.loads(d["metrics"] or "{}")
    p = data_dir() / "reviews" / f"{d['id']}.npy"
    d["mask"] = np.load(p) if p.exists() else None
    return d


def list_reviews():
    with conn() as c:
        rows = c.execute("SELECT r.*, s.field_id FROM reviews r JOIN scans s ON s.id=r.scan_id ORDER BY r.id").fetchall()
    out = []
    for r in rows:
        d = dict(r); d["metrics"] = json.loads(d["metrics"] or "{}"); out.append(d)
    return out


# ------------------------------------------------------------------ photos (RGB analyses)
_PHOTO_Q = ("SELECT p.*, f.name AS field_name, f.crop, f.area_ha, f.owner_id, u.name AS farmer_name, "
            "r.name AS reviewer_name FROM photos p JOIN fields f ON f.id=p.field_id JOIN users u ON u.id=f.owner_id "
            "LEFT JOIN users r ON r.id=p.reviewer_id ")


def _photo(r):
    d = dict(r); d["details"] = json.loads(d["details"] or "{}"); return d


def save_photo(field_id, rgb, zone, details, note="", request_review=True):
    from PIL import Image
    with conn() as c:
        pid = c.execute("INSERT INTO photos(field_id,created,note,details,review_status) VALUES(?,?,?,?,?)",
                        (field_id, now(), note, json.dumps(details), "pending" if request_review else "not_requested")).lastrowid
    (data_dir() / "photos").mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(data_dir() / "photos" / f"{pid}.jpg", quality=90)
    np.savez_compressed(data_dir() / "photos" / f"{pid}.npz", zone=zone.astype(np.uint8))
    return pid


def list_photos(field_id=None, owner_id=None, status=None):
    where, args = [], []
    if field_id is not None: where.append("p.field_id=?"); args.append(field_id)
    if owner_id is not None: where.append("f.owner_id=?"); args.append(owner_id)
    if status is not None: where.append("p.review_status=?"); args.append(status)
    q = _PHOTO_Q + (("WHERE " + " AND ".join(where)) if where else "") + " ORDER BY p.created DESC, p.id DESC"
    with conn() as c:
        return [_photo(r) for r in c.execute(q, args)]


def load_photo_arrays(photo_id):
    from PIL import Image
    rgb = np.asarray(Image.open(data_dir() / "photos" / f"{photo_id}.jpg").convert("RGB"))
    return rgb, np.load(data_dir() / "photos" / f"{photo_id}.npz")["zone"]


def review_photo(photo_id, reviewer_id, diagnosis, status="reviewed"):
    with conn() as c:
        c.execute("UPDATE photos SET diagnosis=?, reviewer_id=?, reviewed=?, review_status=? WHERE id=?",
                  (diagnosis, reviewer_id, now(), status, photo_id))


def request_photo_review(photo_id):
    with conn() as c:
        c.execute("UPDATE photos SET review_status='pending' WHERE id=? AND review_status='not_requested'", (photo_id,))


def delete_photo(photo_id):
    with conn() as c:
        c.execute("DELETE FROM comments WHERE kind='photo' AND ref_id=?", (photo_id,))
        c.execute("DELETE FROM photos WHERE id=?", (photo_id,))
    for ext in ("jpg", "npz"):
        (data_dir() / "photos" / f"{photo_id}.{ext}").unlink(missing_ok=True)


# ------------------------------------------------------------------ comments & journal
def add_comment(kind, ref_id, user_id, text):
    with conn() as c:
        c.execute("INSERT INTO comments(kind,ref_id,user_id,created,text) VALUES(?,?,?,?,?)",
                  (kind, ref_id, user_id, now(), text.strip()))


def list_comments(kind, ref_id):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT m.*, u.name, u.role FROM comments m JOIN users u ON u.id=m.user_id "
            "WHERE m.kind=? AND m.ref_id=? ORDER BY m.id", (kind, ref_id))]


def add_journal(field_id, user_id, day, action, notes):
    with conn() as c:
        c.execute("INSERT INTO journal(field_id,user_id,day,action,notes,created) VALUES(?,?,?,?,?,?)",
                  (field_id, user_id, day, action, notes, now()))


def list_journal(field_id):
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM journal WHERE field_id=? ORDER BY day DESC, id DESC", (field_id,))]


def delete_journal(entry_id, field_ids):
    with conn() as c:
        for fid in field_ids:
            c.execute("DELETE FROM journal WHERE id=? AND field_id=?", (entry_id, fid))
