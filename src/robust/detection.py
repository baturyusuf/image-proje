"""
Robust Wavelet Watermarking (Extraction / Verification)

Size-agnostic adaptation of your detection.py:
- removes hardcoded 16x16/32x32 reshapes and loops
- adds semi-blind reference support for ".npz" metadata flow
"""

import os
import math
from functools import lru_cache
from mpmath import csc
import numpy as np
import pywt
from scipy.signal import convolve2d
from math import sqrt

BLOCK_SIZE = 4
ALPHA = 2
THRESHOLD = 11.79
WAVELET = "haar"
MARK_SIZE = 1024


def _crop_to_multiple(arr2d: np.ndarray, multiple: int) -> np.ndarray:
    h, w = arr2d.shape
    h2 = (h // multiple) * multiple
    w2 = (w // multiple) * multiple
    return arr2d[:h2, :w2]


def quantize(imm, block_size=BLOCK_SIZE):
    imm = np.asarray(imm)
    if imm.ndim != 2:
        raise ValueError("quantize expects a 2D array.")
    h, w = imm.shape
    h2 = (h // block_size) * block_size
    w2 = (w // block_size) * block_size
    imm = imm[:h2, :w2]
    nb_y = h2 // block_size
    nb_x = w2 // block_size
    return np.reshape(imm, (nb_y, nb_x, block_size, block_size))


def apdcbt(imm):
    M, N = imm.shape
    V = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            if n == 0:
                V[m][n] = (N - m) / (N ** 2)
            else:
                V[m][n] = (
                    ((N - m) * math.cos((m * n * math.pi) / N) - csc((n * math.pi) / N) * math.sin((m * n * math.pi) / N))
                    / (N ** 2)
                )
    Y = V @ imm @ V.T
    return Y


def get_coefficient_matrix(image_block):
    nb_y, nb_x = image_block.shape[0], image_block.shape[1]
    for i in range(nb_y):
        for j in range(nb_x):
            image_block[i][j] = apdcbt(image_block[i][j])
    return image_block


def merge_watermarks(w1, w2, t=.5):
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    watermark = (w1 + w2) / 2.0
    out = np.zeros_like(watermark, dtype=np.float64)
    out[watermark >= t] = 1.0
    return out


def similarity(X, X_star):
    X = np.asarray(X).reshape((-1,))
    X_star = np.asarray(X_star).reshape((-1,))
    denom = np.sqrt(np.sum(np.multiply(X_star, X_star)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.multiply(X, X_star)) / denom)


def check_wm(watermark_originale, watermark_attacked, threshold: float = THRESHOLD):
    watermark_originale = np.reshape(watermark_originale, (MARK_SIZE,))
    watermark_attacked = np.reshape(watermark_attacked, (MARK_SIZE,))
    if watermark_attacked.any() == 1:
        sim = similarity(watermark_originale, watermark_attacked)
    else:
        sim = 0.0
    return 1 if sim > threshold else 0


def extract_watermark(image, watermarked, alpha=ALPHA):
    """
    Original signature preserved (non-blind; needs cover image).
    """
    image = np.asarray(image)
    watermarked = np.asarray(watermarked)
    if image.ndim != 2 or watermarked.ndim != 2:
        raise ValueError("extract_watermark expects 2D grayscale arrays.")

    h = min(image.shape[0], watermarked.shape[0])
    w = min(image.shape[1], watermarked.shape[1])
    image = image[:h, :w]
    watermarked = watermarked[:h, :w]
    image = _crop_to_multiple(image, 16)
    watermarked = _crop_to_multiple(watermarked, 16)

    LL, _ = pywt.dwt2(image, WAVELET)
    LL2, (LH2, HL2, _) = pywt.dwt2(LL, WAVELET)

    LL_w, _ = pywt.dwt2(watermarked, WAVELET)
    LL2_w, (LH2_w, HL2_w, _) = pywt.dwt2(LL_w, WAVELET)

    LH2_b = quantize(LH2, BLOCK_SIZE)
    HL2_b = quantize(HL2, BLOCK_SIZE)
    LH2_wb = quantize(LH2_w, BLOCK_SIZE)
    HL2_wb = quantize(HL2_w, BLOCK_SIZE)

    coefficient_matrix_LH2 = get_coefficient_matrix(LH2_b)
    coefficient_matrix_HL2 = get_coefficient_matrix(HL2_b)
    coefficient_matrix_LH2_w = get_coefficient_matrix(LH2_wb)
    coefficient_matrix_HL2_w = get_coefficient_matrix(HL2_wb)

    base_LH = np.abs(coefficient_matrix_LH2).reshape((-1,))
    base_HL = np.abs(coefficient_matrix_HL2).reshape((-1,))
    wm_LH = np.abs(coefficient_matrix_LH2_w).reshape((-1,))
    wm_HL = np.abs(coefficient_matrix_HL2_w).reshape((-1,))

    if base_LH.size - 1 < MARK_SIZE or base_HL.size - 1 < MARK_SIZE:
        raise ValueError("Image is too small in wavelet domain for the configured MARK_SIZE.")

    loc_LH = np.argsort(-base_LH)[1: MARK_SIZE + 1]
    loc_HL = np.argsort(-base_HL)[1: MARK_SIZE + 1]

    w1 = np.zeros(MARK_SIZE, dtype=np.float64)
    w2 = np.zeros(MARK_SIZE, dtype=np.float64)

    for idx, loc in enumerate(loc_LH):
        w1[idx] = (wm_LH[loc] - base_LH[loc]) / alpha

    for idx, loc in enumerate(loc_HL):
        w2[idx] = (wm_HL[loc] - base_HL[loc]) / alpha

    side = int(math.isqrt(MARK_SIZE))
    w1 = np.reshape(w1, (side, side))
    w2 = np.reshape(w2, (side, side))
    return w1, w2


# ----------------------------
# Semi-blind reference (metadata) helpers
# ----------------------------
def compute_reference(image: np.ndarray, block_size: int = BLOCK_SIZE, wavelet: str = WAVELET, mark_size: int = MARK_SIZE):
    """
    Compute locations and reference magnitudes from the COVER image, sufficient for
    semi-blind extraction (no need to keep the full original image later).
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("compute_reference expects a 2D grayscale array.")
    image = _crop_to_multiple(image, 16)

    LL, _ = pywt.dwt2(image, wavelet)
    LL2, (LH2, HL2, _) = pywt.dwt2(LL, wavelet)

    LH2_b = quantize(LH2, block_size)
    HL2_b = quantize(HL2, block_size)

    coef_LH = np.abs(get_coefficient_matrix(LH2_b)).reshape((-1,))
    coef_HL = np.abs(get_coefficient_matrix(HL2_b)).reshape((-1,))

    if coef_LH.size - 1 < mark_size or coef_HL.size - 1 < mark_size:
        raise ValueError("Image is too small in wavelet domain for the requested mark_size.")

    loc_LH = np.argsort(-coef_LH)[1: mark_size + 1].astype(np.int32)
    loc_HL = np.argsort(-coef_HL)[1: mark_size + 1].astype(np.int32)

    ref_LH = coef_LH[loc_LH].astype(np.float32)
    ref_HL = coef_HL[loc_HL].astype(np.float32)

    return {
        "loc_LH2": loc_LH,
        "ref_LH2": ref_LH,
        "loc_HL2": loc_HL,
        "ref_HL2": ref_HL,
        "shape_hw": np.array(image.shape, dtype=np.int32),
        "block_size": int(block_size),
        "wavelet": str(wavelet),
        "mark_size": int(mark_size),
    }


def extract_watermark_from_reference(watermarked: np.ndarray, reference: dict, alpha: float = ALPHA):
    """
    Semi-blind extraction using stored reference (locations + magnitudes), without cover image.
    """
    watermarked = np.asarray(watermarked)
    if watermarked.ndim != 2:
        raise ValueError("extract_watermark_from_reference expects a 2D grayscale array.")

    shape_hw = tuple(int(x) for x in reference["shape_hw"])
    h, w = shape_hw
    watermarked = watermarked[:h, :w]
    watermarked = _crop_to_multiple(watermarked, 16)

    block_size = int(reference["block_size"])
    wavelet = str(reference["wavelet"])
    mark_size = int(reference["mark_size"])

    LL_w, _ = pywt.dwt2(watermarked, wavelet)
    LL2_w, (LH2_w, HL2_w, _) = pywt.dwt2(LL_w, wavelet)

    LH2_wb = quantize(LH2_w, block_size)
    HL2_wb = quantize(HL2_w, block_size)

    coef_LH_w = np.abs(get_coefficient_matrix(LH2_wb)).reshape((-1,))
    coef_HL_w = np.abs(get_coefficient_matrix(HL2_wb)).reshape((-1,))

    loc_LH = np.asarray(reference["loc_LH2"], dtype=np.int64)
    ref_LH = np.asarray(reference["ref_LH2"], dtype=np.float64)
    loc_HL = np.asarray(reference["loc_HL2"], dtype=np.int64)
    ref_HL = np.asarray(reference["ref_HL2"], dtype=np.float64)

    if loc_LH.size != mark_size or loc_HL.size != mark_size:
        raise ValueError("Reference mark_size does not match stored locations length.")

    w1 = (coef_LH_w[loc_LH] - ref_LH) / float(alpha)
    w2 = (coef_HL_w[loc_HL] - ref_HL) / float(alpha)

    side = int(math.isqrt(mark_size))
    w1 = np.reshape(w1, (side, side))
    w2 = np.reshape(w2, (side, side))
    return w1, w2


# ----------------------------
# wPSNR (used for evaluation)
# ----------------------------
@lru_cache(maxsize=1)
def _load_csf_matrix(csf_path: str):
    return np.genfromtxt(csf_path, delimiter=",")


def _ensure_csf(csf_path: str):
    if os.path.isfile(csf_path):
        return
    os.makedirs(os.path.dirname(csf_path), exist_ok=True)

    url = "https://drive.google.com/uc?export=download&id=1w43k1BTfrWm6X0rqAOQhrbX6JDhIKTRW"
    try:
        import urllib.request
        urllib.request.urlretrieve(url, csf_path)
    except Exception as e:
        raise FileNotFoundError(
            f"csf.csv not found at '{csf_path}' and download failed. "
            "Place csf.csv under assets/ manually."
        ) from e


def wpsnr(img1, img2, csf_path: str = os.path.join("assets", "csf.csv")):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0

    difference = img1 - img2
    if not np.any(difference):
        return 9999999.0

    _ensure_csf(csf_path)
    csf = _load_csf_matrix(csf_path)

    ew = convolve2d(difference, np.rot90(csf, 2), mode="valid")
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return float(decibels)
