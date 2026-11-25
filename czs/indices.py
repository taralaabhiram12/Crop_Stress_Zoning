"""Vegetation indices from multispectral bands."""
EPS = 1e-6

INDEX_INFO = {
    "NDVI": ("(NIR-Red)/(NIR+Red)", "Overall greenness / biomass"),
    "GNDVI": ("(NIR-Green)/(NIR+Green)", "Chlorophyll content, less saturation"),
    "NDRE": ("(NIR-RedEdge)/(NIR+RedEdge)", "Early stress / nitrogen status"),
    "SAVI": ("1.5*(NIR-Red)/(NIR+Red+0.5)", "NDVI corrected for soil background"),
    "EVI": ("2.5*(NIR-Red)/(NIR+6Red-7.5Blue+1)", "Enhanced for dense canopy"),
    "CIre": ("NIR/RedEdge - 1", "Red-edge chlorophyll index"),
    "NDWI": ("(Green-NIR)/(Green+NIR)", "Canopy water content proxy"),
    "GCI": ("NIR/Green - 1", "Green chlorophyll index"),
}


def compute_indices(b):
    nir, red, green, re, blue = b["nir"], b["red"], b["green"], b["rededge"], b["blue"]
    return {
        "NDVI": (nir - red) / (nir + red + EPS),
        "GNDVI": (nir - green) / (nir + green + EPS),
        "NDRE": (nir - re) / (nir + re + EPS),
        "SAVI": 1.5 * (nir - red) / (nir + red + 0.5),
        "EVI": 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1),
        "CIre": nir / (re + EPS) - 1,
        "NDWI": (green - nir) / (green + nir + EPS),
        "GCI": nir / (green + EPS) - 1,
    }
