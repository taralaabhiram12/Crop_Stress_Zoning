"""Password hashing, registration and login with basic brute-force lockout."""
import hashlib
import hmac
import os
import re
import time

from . import db
from .config import agronomist_code

ITERATIONS = 200_000
MAX_FAILS, LOCK_SECONDS = 5, 300


def hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return h.hex(), salt.hex()


def verify_password(password, pw_hash, salt_hex):
    h, _ = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(h, pw_hash)


def register(username, name, password, role, invite=""):
    username, name = username.strip().lower(), name.strip()
    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        return False, "Username: 3–30 characters, letters, digits or underscore."
    if len(name) < 2:
        return False, "Please enter your full name."
    if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return False, "Password must be at least 8 characters and contain a letter and a digit."
    if role not in ("farmer", "agronomist"):
        return False, "Invalid role."
    if role == "agronomist" and not hmac.compare_digest(invite.strip(), agronomist_code()):
        return False, "Invalid agronomist invitation code."
    h, s = hash_password(password)
    try:
        db.create_user(username, name, role, h, s)
    except ValueError as e:
        return False, str(e)
    return True, "Account created. You can log in now."


def authenticate(username, password):
    u = db.get_user(username.strip().lower())
    if u is None:
        hash_password(password)  # equalise timing
        return None, "Invalid username or password."
    if u["locked_until"] and u["locked_until"] > time.time():
        mins = int((u["locked_until"] - time.time()) // 60) + 1
        return None, f"Too many failed attempts. Try again in about {mins} min."
    if verify_password(password, u["pw_hash"], u["salt"]):
        db.set_login_state(u["id"], 0, 0.0)
        return {k: u[k] for k in ("id", "username", "name", "role")}, "ok"
    fails = u["fail_count"] + 1
    if fails >= MAX_FAILS:
        db.set_login_state(u["id"], 0, time.time() + LOCK_SECONDS)
    else:
        db.set_login_state(u["id"], fails, 0.0)
    return None, "Invalid username or password."
