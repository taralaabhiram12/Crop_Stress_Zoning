# 🌾 K-Means Crop Stress Zoning – Farmer & Agronomist Web App
**SIH26001** · Tarala Abhiram · RK3P24UKA01
*K-Means clustering-based crop stress zoning using multispectral vegetation indices with agronomist-validated boundaries.*

## ✨ v4 highlights
* **📷 Photo Analysis** – upload ordinary phone/drone photos (single or batch of up to 10) and get a **crop health score (0–100)**, % healthy / yellowing / browning,
  canopy cover, **lesion count**, largest dead patch, a **hotspot grid** (which part of the field to scout first), colour evidence charts,
  automatic findings, annotated PNG, HTML/PDF report. Works without multispectral cameras. Includes sample images to try instantly.
* **How the photo model works** (`czs/rgb.py`): plant mask → visible-band indices (ExG, VARI, GLI, NGRDI, TGI) → **K-Means** on colour/index features →
  each cluster is named from its *real* centroid colour (green / yellow / brown), so a healthy photo is never forced into stress classes.
  Brown ground can be treated as Auto / dead crop / soil. Colour screening only – it cannot name the exact disease.
* **Agronomist Photo Reviews** – queue sorted by worst health, diagnosis box, conversation thread with the farmer.
* **💬 Messages** between farmer and agronomist on photos and scans · **📒 Field Journal** (irrigation, spraying, fertiliser…) shown on the trend chart.
* New look: banner headers, health gauges, cards, sidebar pills.

## What it does
| Project objective | Where it lives |
|---|---|
| 1. Extract vegetation-index / spectral-band features | `czs/indices.py` (8 indices), feature selection in New Scan & Analysis Lab |
| 2. K-Means zoning into healthy / stressed / diseased | `czs/clustering.py`, `czs/pipeline.py`, elbow + silhouette in the Lab |
| 3. Validate boundaries against agronomist markings | **Validate Scan** page (draw/correct zones) + `czs/validation.py` (accuracy, κ, IoU, boundary-F1) |

## Roles
**👨‍🌾 Farmer** – register fields · run scans (simulated drone survey or upload GeoTIFF/.npy) · zone maps with area in hectares ·
send scans to an agronomist · see **validated boundaries, notes and recommendations** · trend charts over time ·
before/after **change-detection** map · alerts when diseased area is high · rule-based advice · downloadable HTML/PDF report and PNG.

**🔬 Agronomist** – review queue prioritised by disease % · **mark/correct zone boundaries** (rectangle/circle tools, smooth, undo,
or upload a mask) with live agreement metrics · validated / re-survey outcome with notes and recommendation ·
**Auto-Tune** (searches k, index set, smoothing to best match validated boundaries) · all-fields hotspot view ·
model-vs-agronomist analytics (aggregate confusion matrix, κ trend) · full **Analysis Lab**.

Security: PBKDF2-SHA256 salted password hashes, account lock-out after 5 failed logins (5 min), farmers can only see their own fields,
agronomist sign-up requires an invitation code.

## Run locally
```bash
pip install -r requirements-dev.txt   # or requirements.txt for runtime only
streamlit run app.py                 # http://localhost:8501
python -m pytest -q tests
```
On first start (with `SEED_DEMO` not `0`) demo accounts are created:
`demo_farmer / Farmer@123` and `demo_agronomist / Agro@1234` (agronomist invite code default: `AGRO-DEMO`).

## Environment variables
| Variable | Purpose | Default |
|---|---|---|
| `DATA_DIR` | Where the SQLite DB and scan files are stored | `./data` |
| `AGRONOMIST_INVITE_CODE` | Code required to register as agronomist | `AGRO-DEMO` ⚠️ change in production |
| `SEED_DEMO` | `0` disables demo accounts/data | `1` locally, `0` in Docker |
| `PORT` | Port (Docker/Heroku-style hosts) | `8501` |

## Deploy
**Docker (any VPS / cloud VM)**
```bash
# edit AGRONOMIST_INVITE_CODE in docker-compose.yml first
docker compose up -d --build         # http://SERVER:8501, data persisted in the "cropdata" volume
```
**Render.com** – push to GitHub → *New → Blueprint* (uses `render.yaml`, Docker + 1 GB persistent disk). Put HTTPS in front (Render does this automatically).
**Railway / Fly.io** – deploy the Dockerfile and mount a volume at `/data`.
**Streamlit Community Cloud** – works for demos (select `app.py`), but its disk is *ephemeral*: accounts and scans reset on restart.
For permanent data use Docker/Render with a volume.

Production checklist: set a strong `AGRONOMIST_INVITE_CODE`, `SEED_DEMO=0`, serve over HTTPS, back up the `/data` volume.
Scaling note: SQLite is fine for a pilot/single instance; migrate `czs/db.py` to PostgreSQL for many concurrent users.

## Using real data
New Scan → *Upload multispectral raster* → multi-band GeoTIFF or `.npy/.npz` (H×W×Bands), map the Blue/Green/Red/Red-Edge/NIR bands.
For hyperspectral cubes, pick the bands closest to those wavelengths. Agronomists can upload field-marking masks (0/1/2) on the Validate page.
The demo imagery is synthetic – results on real data must be judged on real validated scans.

## Layout
```
app.py                 entry point (auth gate + role navigation)
czs/                   core: indices, clustering, validation, db, auth, advice, report, pipeline, seed
ui/                    login, farmer pages, agronomist pages, analysis lab
tests/                 pytest (core, RGB engine, auth, DB, UI end-to-end)
Dockerfile · docker-compose.yml · render.yaml · Procfile · .streamlit/config.toml
```
