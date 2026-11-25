"""Runtime configuration (read from environment at call time)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALERT_DISEASED_PCT = 15.0   # farmer alert threshold (% of field classed Diseased)


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", ROOT / "data"))


def agronomist_code() -> str:
    return os.environ.get("AGRONOMIST_INVITE_CODE", "AGRO-DEMO")


def seed_demo() -> bool:
    return os.environ.get("SEED_DEMO", "1") != "0"
