"""
Attacks for robustness testing.

Adapted from your original attacks.py, but:
- removed top-level side effects (no file IO, no auto-running)
- uses in-memory JPEG compression (no tmp.jpg)
- ensures outputs are uint8 arrays
- keeps the original attack function signatures
"""

import io
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale


def _to_uint8(img):
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def awgn(img, std, seed):
    mean = 0.0
    rng = np.random.default_rng(int(seed))
    attacked = np.asarray(img, dtype=np.float64) + rng.normal(mean, float(std), np.asarray(img).shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked.astype(np.uint8)


def blur(img, sigma):
    attacked = gaussian_filter(np.asarray(img, dtype=np.float64), sigma)
    return _to_uint8(attacked)


def sharpening(img, sigma, alpha):
    img_f = np.asarray(img, dtype=np.float64)
    filter_blurred_f = gaussian_filter(img_f, sigma)
    attacked = img_f + float(alpha) * (img_f - filter_blurred_f)
    return _to_uint8(attacked)


def median(img, kernel_size):
    attacked = medfilt(np.asarray(img, dtype=np.float64), kernel_size)
    return _to_uint8(attacked)


def resizing(img, scale):
    img = np.asarray(img, dtype=np.float64)
    x, y = img.shape
    attacked = rescale(img, float(scale), preserve_range=True, anti_aliasing=True)
    attacked = rescale(attacked, 1.0 / float(scale), preserve_range=True, anti_aliasing=True)
    attacked = attacked[:x, :y]
    return _to_uint8(attacked)


def jpeg_compression(img, QF):
    img = _to_uint8(img)
    pil = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(QF))
    buf.seek(0)
    attacked = Image.open(buf).convert("L")
    return np.asarray(attacked, dtype=np.uint8)


ATTACK_REGISTRY = {
    "AWGN": {
        "fn": awgn,
        "params": {"std": 10.0, "seed": 123},
    },
    "Blur": {
        "fn": blur,
        "params": {"sigma": 1.0},
    },
    "Sharpening": {
        "fn": sharpening,
        "params": {"sigma": 1.0, "alpha": 1.0},
    },
    "Median": {
        "fn": median,
        "params": {"kernel_size": 3},
    },
    "Resizing": {
        "fn": resizing,
        "params": {"scale": 0.5},
    },
    "JPEG": {
        "fn": jpeg_compression,
        "params": {"QF": 50},
    },
}


def apply_attack(img, attack_name: str, **kwargs):
    """
    Convenience wrapper for the Streamlit layer.
    img must be 2D grayscale (Y channel).
    """
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack_name}")

    fn = ATTACK_REGISTRY[attack_name]["fn"]
    params = dict(ATTACK_REGISTRY[attack_name]["params"])
    params.update(kwargs)

    # Map params to signature without changing original function shapes
    if fn is awgn:
        return fn(img, params["std"], params["seed"])
    if fn is blur:
        return fn(img, params["sigma"])
    if fn is sharpening:
        return fn(img, params["sigma"], params["alpha"])
    if fn is median:
        return fn(img, params["kernel_size"])
    if fn is resizing:
        return fn(img, params["scale"])
    if fn is jpeg_compression:
        return fn(img, params["QF"])

    return fn(img, **params)
