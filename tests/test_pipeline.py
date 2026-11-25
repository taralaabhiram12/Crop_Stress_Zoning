import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from czs.synthetic import generate_field
from czs.indices import compute_indices
from czs.clustering import build_features, kmeans_zones
from czs.validation import metrics, boundary_f1


def _run(k=3):
    f = generate_field(96, 1)
    X, _, shp = build_features(compute_indices(f["bands"]), f["bands"], ["NDVI", "NDRE"])
    cls, *_ = kmeans_zones(X, shp, k=k)
    return f, cls


def test_ndvi_range():
    f = generate_field(64, 0)
    nd = compute_indices(f["bands"])["NDVI"]
    assert nd.min() > -1 and nd.max() < 1


def test_zones_ordered_by_health():
    f, cls = _run()
    nd = compute_indices(f["bands"])["NDVI"]
    assert nd[cls == 0].mean() > nd[cls == 1].mean() > nd[cls == 2].mean()


def test_agreement_with_agronomist():
    f, cls = _run()
    m = metrics(cls, f["agronomist"])
    assert m["overall_accuracy"] > 0.75 and m["kappa"] > 0.6
    assert boundary_f1(cls, f["agronomist"])["f1"] > 0.4


def test_other_k_valid_classes():
    for k in (2, 4, 5):
        _, cls = _run(k)
        assert set(np.unique(cls)) <= {0, 1, 2}
