"""Headless pipeline: python run_pipeline.py --k 3 --out results"""
import argparse, json, os
import numpy as np
from PIL import Image
from czs import CLASS_COLORS, CLASS_NAMES
from czs.synthetic import generate_field
from czs.indices import compute_indices
from czs.clustering import build_features, kmeans_zones
from czs.validation import metrics, boundary_f1

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, default=3)
p.add_argument("--size", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--features", nargs="+", default=["NDVI", "GNDVI", "NDRE", "SAVI"])
p.add_argument("--out", default="results")
a = p.parse_args()

f = generate_field(a.size, a.seed)
X, names, shape = build_features(compute_indices(f["bands"]), f["bands"], a.features)
cls, _, _, _ = kmeans_zones(X, shape, k=a.k, ref_col=0)
m = metrics(cls, f["agronomist"]); b = boundary_f1(cls, f["agronomist"])
os.makedirs(a.out, exist_ok=True)
rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in CLASS_COLORS], np.uint8)
Image.fromarray(rgb[cls]).resize((512, 512), Image.NEAREST).save(f"{a.out}/zones.png")
Image.fromarray(rgb[f["agronomist"]]).resize((512, 512), Image.NEAREST).save(f"{a.out}/agronomist.png")
rep = dict(features=names, k=a.k, overall_accuracy=m["overall_accuracy"], kappa=m["kappa"],
           mIoU=m["miou"], boundary_f1=b["f1"],
           zone_area_pct={CLASS_NAMES[i]: float((cls == i).mean() * 100) for i in range(3)})
json.dump(rep, open(f"{a.out}/report.json", "w"), indent=2)
print(json.dumps(rep, indent=2))
