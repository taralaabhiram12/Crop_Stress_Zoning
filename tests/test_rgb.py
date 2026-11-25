import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from czs.rgb import analyse, advise_rgb, overlay, BG
from czs.synthetic import rgb_demo


def _flat(color, noise=6, n=160):
    g = np.zeros((n, n, 3)); g[...] = color
    return (g + np.random.default_rng(0).normal(0, noise, g.shape)).clip(0, 255).astype(np.uint8)


def test_healthy_photo_stays_healthy():
    d = analyse(_flat((60, 140, 50)), "closeup")["details"]
    assert d["zone_pct"][0] > 98 and d["grade"] == "Excellent"


def test_dead_photo_is_critical():
    d = analyse(_flat((120, 78, 40)), "closeup")["details"]
    assert d["zone_pct"][2] > 98 and d["grade"] == "Critical"


def test_yellow_photo_is_stressed():
    d = analyse(_flat((200, 180, 60)), "closeup")["details"]
    assert d["zone_pct"][1] > 90


def test_leaf_lesions_detected():
    d = analyse(rgb_demo("leaf"), "closeup")["details"]
    assert 10 <= d["lesion_count"] <= 22 and d["zone_pct"][2] > 2


def test_aerial_dead_patch_and_soil_handling():
    im = rgb_demo("aerial", 360, 1)
    soil = analyse(im, "aerial", brown="soil")["details"]
    auto = analyse(im, "aerial")["details"]
    assert auto["zone_pct"][2] > soil["zone_pct"][2] + 10 and auto["canopy_pct"] > 85
    assert auto["largest_patch_pct"] > 10
    assert any(p == "High" for p, *_ in advise_rgb(auto))


def test_grid_and_overlay_shapes():
    im = rgb_demo("aerial"); r = analyse(im, "aerial", grid=4)
    assert len(r["details"]["grid"]) == 4 and overlay(im, r["zone"]).shape == im.shape
    assert set(np.unique(r["zone"])) <= {0, 1, 2, BG}


def test_empty_image_raises():
    with pytest.raises(ValueError):
        analyse(_flat((235, 235, 235), 1), "closeup")
