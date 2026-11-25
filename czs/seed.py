"""Create demo accounts + a demo field/scan on a fresh database (disable with SEED_DEMO=0)."""
from . import auth, db
from .config import seed_demo
from .pipeline import cube_from_bands, run_analysis
from .synthetic import generate_field


def seed_if_empty():
    if db.count_users() or not seed_demo():
        return
    auth.register("demo_farmer", "Demo Farmer", "Farmer@123", "farmer")
    auth.register("demo_agronomist", "Dr. Demo Agronomist", "Agro@1234", "agronomist", auth.agronomist_code())
    farmer = db.get_user("demo_farmer")
    fid = db.add_field(farmer["id"], "North Paddock", "Wheat", 12.0, "Phagwara, Punjab")
    d = generate_field(128, 42)
    r = run_analysis(d["bands"])
    db.save_scan(fid, cube_from_bands(d["bands"]), r["cls"], r["params"], r["stats"], "simulated", True)
