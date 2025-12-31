# robust_wavelet.py
# Robust ID watermarking (Wavelet + block APDCBT) - cleaned & Streamlit-friendly

from __future__ import annotations

import math
import hashlib
from functools import lru_cache
from typing import Tuple

import numpy as np
import pywt
from PIL import Image


# -----------------------------
# Utilities
# -----------------------------

def _bytes_to_bits(b: bytes) -> list[int]:
    bits = []
    for byte in b:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits


def generate_mark_from_id(secret_key: str, watermark_id: str, mark_size: int = 1024) -> np.ndarray:
    """
    ID + secret_key -> deterministic 0/1 watermark sequence of length mark_size.
    Note: This is "verification by expected ID". It is not designed to *decode* ID without knowing it.
    """
    payload = f"{secret_key}|{watermark_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()           # 256 bits
    bits256 = _bytes_to_bits(digest)                    # length 256 (0/1)
    reps = (mark_size + len(bits256) - 1) // len(bits256)
    bits = (bits256 * reps)[:mark_size]                 # length mark_size
    return np.array(bits, dtype=np.float64)             # float for math ops


def _crop_to_multiple(img: Image.Image, multiple: int = 16) -> Image.Image:
    w, h = img.size
    w2 = (w // multiple) * multiple
    h2 = (h // multiple) * multiple
    if w2 <= 0 or h2 <= 0:
        raise ValueError("Image too small for robust watermarking.")
    return img.crop((0, 0, w2, h2))


def _to_y_channel(img: Image.Image) -> Tuple[np.ndarray, Image.Image]:
    """
    Returns:
      y: float64 array in [0,255]
      ycbcr_img: PIL image in YCbCr mode (for reconstruction)
    """
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y_arr = np.array(y, dtype=np.float64)
    return y_arr, ycbcr


def _from_y_channel(ycbcr_img: Image.Image, new_y: np.ndarray) -> Image.Image:
    new_y_u8 = np.clip(new_y, 0, 255).astype(np.uint8)
    y_img = Image.fromarray(new_y_u8, mode="L")
    _, cb, cr = ycbcr_img.split()
    merged = Image.merge("YCbCr", (y_img, cb, cr)).convert("RGB")
    return merged


# -----------------------------
# APDCBT block transform (cached)
# -----------------------------

@lru_cache(maxsize=None)
def _apdcbt_mats(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds V and V^{-1} for APDCBT of size n.
    csc(x) is implemented as 1/sin(x) (no mpmath dependency).
    """
    V = np.zeros((n, n), dtype=np.float64)
    for m in range(n):
        for k in range(n):
            if k == 0:
                V[m, k] = (n - m) / (n**2)
            else:
                angle = (k * math.pi) / n
                csc_val = 1.0 / math.sin(angle)
                V[m, k] = (
                    (n - m) * math.cos((m * k * math.pi) / n)
                    - csc_val * math.sin((m * k * math.pi) / n)
                ) / (n**2)

    Vinv = np.linalg.inv(V)
    return V, Vinv


def _apdcbt_block(block: np.ndarray, V: np.ndarray) -> np.ndarray:
    return V @ block @ V.T


def _iapdcbt_block(coeff_block: np.ndarray, Vinv: np.ndarray) -> np.ndarray:
    return Vinv @ coeff_block @ Vinv.T


# -----------------------------
# Block helpers
# -----------------------------

def _image_to_blocks(img2d: np.ndarray, block_size: int) -> np.ndarray:
    h, w = img2d.shape
    if (h % block_size) != 0 or (w % block_size) != 0:
        raise ValueError("Subband size must be divisible by block_size.")
    bh, bw = h // block_size, w // block_size
    # (bh, block, bw, block) -> (bh, bw, block, block)
    return img2d.reshape(bh, block_size, bw, block_size).swapaxes(1, 2).copy()


def _blocks_to_image(blocks: np.ndarray) -> np.ndarray:
    # (bh, bw, block, block) -> (bh, block, bw, block) -> (h,w)
    bh, bw, bs, _ = blocks.shape
    return blocks.swapaxes(1, 2).reshape(bh * bs, bw * bs)


def _get_coeff_matrix(sub_band: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      coeff_img: 2D coefficient "image" after APDCBT applied per block
      V, Vinv: cached matrices for transform size block_size
    """
    V, Vinv = _apdcbt_mats(block_size)
    blocks = _image_to_blocks(sub_band, block_size)

    bh, bw, _, _ = blocks.shape
    for i in range(bh):
        for j in range(bw):
            blocks[i, j] = _apdcbt_block(blocks[i, j], V)

    coeff_img = _blocks_to_image(blocks)
    return coeff_img, V, Vinv


def _hide_mark(coeff_img: np.ndarray, mark_bits: np.ndarray, alpha: float) -> np.ndarray:
    """
    Embeds mark bits into the largest-magnitude coefficients (skipping the very largest).
    """
    flat = coeff_img.reshape(-1)
    sgn = np.where(flat >= 0, 1.0, -1.0)
    mod = np.abs(flat)

    order = np.argsort(-mod)  # descending magnitude
    n = min(len(mark_bits), len(order) - 1)  # skip index 0

    mod_w = mod.copy()
    for idx, loc in enumerate(order[1:n + 1]):
        mod_w[loc] = mod[loc] + alpha * float(mark_bits[idx])

    out = (mod_w * sgn).reshape(coeff_img.shape)
    return out


def _embed_into_subband(sub_band: np.ndarray, mark_bits: np.ndarray, alpha: float, block_size: int) -> np.ndarray:
    coeff_img, V, Vinv = _get_coeff_matrix(sub_band, block_size)
    embedded_coeff = _hide_mark(coeff_img, mark_bits, alpha)

    # Inverse APDCBT per block
    blocks = _image_to_blocks(embedded_coeff, block_size)
    bh, bw, _, _ = blocks.shape
    for i in range(bh):
        for j in range(bw):
            blocks[i, j] = _iapdcbt_block(blocks[i, j], Vinv)

    return _blocks_to_image(blocks)


# -----------------------------
# Public API
# -----------------------------

def embed_robust_id(
    img_rgb: Image.Image,
    watermark_id: str,
    secret_key: str,
    alpha: float = 2.0,
    block_size: int = 4,
    wavelet: str = "haar",
    mark_size: int = 1024,
) -> Image.Image:
    """
    Robust watermark embedding on luminance (Y) channel.
    """
    img_rgb = _crop_to_multiple(img_rgb.convert("RGB"), multiple=16)
    y_arr, ycbcr = _to_y_channel(img_rgb)

    # 2-level DWT
    LL, (LH, HL, HH) = pywt.dwt2(y_arr, wavelet)
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL, wavelet)

    # Ensure LH2 / HL2 are divisible by block_size
    h2, w2 = LH2.shape
    h2c = (h2 // block_size) * block_size
    w2c = (w2 // block_size) * block_size
    LH2 = LH2[:h2c, :w2c]
    HL2 = HL2[:h2c, :w2c]
    HH2 = HH2[:h2c, :w2c]
    LL2 = LL2[:h2c, :w2c]

    mark_bits = generate_mark_from_id(secret_key, watermark_id, mark_size=mark_size)

    LH2_w = _embed_into_subband(LH2, mark_bits, alpha=alpha, block_size=block_size)
    HL2_w = _embed_into_subband(HL2, mark_bits, alpha=alpha, block_size=block_size)

    LL_w = pywt.idwt2((LL2, (LH2_w, HL2_w, HH2)), wavelet)
    y_w = pywt.idwt2((LL_w, (LH, HL, HH)), wavelet)

    # Match size back to y_arr
    y_w = y_w[: y_arr.shape[0], : y_arr.shape[1]]

    out = _from_y_channel(ycbcr, y_w)
    return out


def extract_robust_bits(
    original_rgb: Image.Image,
    suspect_rgb: Image.Image,
    alpha: float = 2.0,
    block_size: int = 4,
    wavelet: str = "haar",
    mark_size: int = 1024,
    merge_threshold: float = 0.5,
) -> np.ndarray:
    """
    Non-blind extraction: requires original cover image.
    Returns 32x32 binary watermark (uint8).
    """
    original_rgb = _crop_to_multiple(original_rgb.convert("RGB"), multiple=16)
    suspect_rgb = _crop_to_multiple(suspect_rgb.convert("RGB"), multiple=16)

    # Align sizes to min common region
    w = min(original_rgb.size[0], suspect_rgb.size[0])
    h = min(original_rgb.size[1], suspect_rgb.size[1])
    w = (w // 16) * 16
    h = (h // 16) * 16
    original_rgb = original_rgb.crop((0, 0, w, h))
    suspect_rgb = suspect_rgb.crop((0, 0, w, h))

    y0, _ = _to_y_channel(original_rgb)
    y1, _ = _to_y_channel(suspect_rgb)

    # DWTs
    LL0, (LH0, HL0, HH0) = pywt.dwt2(y0, wavelet)
    LL02, (LH02, HL02, HH02) = pywt.dwt2(LL0, wavelet)

    LL1, (LH1, HL1, HH1) = pywt.dwt2(y1, wavelet)
    LL12, (LH12, HL12, HH12) = pywt.dwt2(LL1, wavelet)

    # crop subbands to multiples
    h2, w2 = LH02.shape
    h2c = (h2 // block_size) * block_size
    w2c = (w2 // block_size) * block_size

    LH02 = LH02[:h2c, :w2c]
    HL02 = HL02[:h2c, :w2c]
    LH12 = LH12[:h2c, :w2c]
    HL12 = HL12[:h2c, :w2c]

    # Coeff matrices
    C_LH0, _, _ = _get_coeff_matrix(LH02, block_size)
    C_HL0, _, _ = _get_coeff_matrix(HL02, block_size)

    C_LH1, _, _ = _get_coeff_matrix(LH12, block_size)
    C_HL1, _, _ = _get_coeff_matrix(HL12, block_size)

    def _extract_from_pair(C0: np.ndarray, C1: np.ndarray) -> np.ndarray:
        C0a = np.abs(C0).reshape(-1)
        order = np.argsort(-C0a)
        C1a = np.abs(C1).reshape(-1)

        n = min(mark_size, len(order) - 1)
        wv = np.zeros(n, dtype=np.float64)
        for idx, loc in enumerate(order[1:n + 1]):
            wv[idx] = (C1a[loc] - C0a[loc]) / float(alpha)
        return wv

    w1 = _extract_from_pair(C_LH0, C_LH1)
    w2 = _extract_from_pair(C_HL0, C_HL1)

    n = min(len(w1), len(w2))
    w_soft = (w1[:n] + w2[:n]) / 2.0
    w_soft = np.clip(w_soft, 0.0, 1.0)

    bits = (w_soft >= merge_threshold).astype(np.uint8)

    # Return 32x32 view (pad/truncate to 1024)
    out = np.zeros(1024, dtype=np.uint8)
    out[: min(1024, len(bits))] = bits[: min(1024, len(bits))]
    return out.reshape(32, 32)


def similarity(expected_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """
    Same spirit as provided code (normalized correlation-like score).
    """
    X = expected_bits.reshape(-1).astype(np.float64)
    Xs = extracted_bits.reshape(-1).astype(np.float64)
    denom = float(np.sqrt(np.sum(Xs * Xs)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(X * Xs) / denom)


def check_robust_id(
    original_rgb: Image.Image,
    suspect_rgb: Image.Image,
    watermark_id: str,
    secret_key: str,
    alpha: float = 2.0,
    threshold: float = 11.79,
    block_size: int = 4,
    wavelet: str = "haar",
) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """
    Returns: (found, sim, expected_32x32, extracted_32x32)
    """
    expected = generate_mark_from_id(secret_key, watermark_id, mark_size=1024).reshape(32, 32).astype(np.uint8)
    extracted = extract_robust_bits(
        original_rgb=original_rgb,
        suspect_rgb=suspect_rgb,
        alpha=alpha,
        block_size=block_size,
        wavelet=wavelet,
        mark_size=1024,
    )
    sim = similarity(expected, extracted)
    found = sim > threshold
    return found, sim, expected, extracted
