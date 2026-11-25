"""Loading real multispectral data and agronomist masks."""
import io
import numpy as np
from PIL import Image


def load_cube(file_bytes, name):
    """Return float array (H, W, B) from .npy / .npz / GeoTIFF."""
    n = name.lower()
    if n.endswith(".npy"):
        arr = np.load(io.BytesIO(file_bytes))
    elif n.endswith(".npz"):
        z = np.load(io.BytesIO(file_bytes)); arr = z[list(z.keys())[0]]
    else:
        import tifffile
        arr = tifffile.imread(io.BytesIO(file_bytes))
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:  # (B,H,W) -> (H,W,B)
            arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValueError("Expected a multi-band raster (H x W x Bands).")
    return arr.astype(np.float32)


def load_mask(file_bytes, name, shape):
    """Agronomist mask: values 0/1/2 (healthy/stressed/diseased) or grey levels."""
    if name.lower().endswith(".npy"):
        m = np.load(io.BytesIO(file_bytes))
    else:
        m = np.array(Image.open(io.BytesIO(file_bytes)).convert("L"))
    if m.shape != shape:
        m = np.array(Image.fromarray(m.astype(np.uint8)).resize(shape[::-1], Image.NEAREST))
    if m.max() > 2:
        m = np.digitize(m, [85, 170])
    return m.astype(np.uint8)


def make_demo_files(out_dir):
    from .synthetic import generate_field
    f = generate_field(size=96, seed=7)
    cube = np.stack([f["bands"][b] for b in ["blue", "green", "red", "rededge", "nir"]], -1)
    np.save(f"{out_dir}/demo_cube.npy", cube)
    np.save(f"{out_dir}/demo_agronomist_mask.npy", f["agronomist"])
