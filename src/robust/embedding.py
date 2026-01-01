"""
Robust Wavelet Watermarking (Embedding)

Size-agnostic adaptation of your embedding.py:
- removes hardcoded 32x32 reshapes / loops
- computes block grid from actual sub-band dimensions
- crops safely to multiples needed by DWT + block processing
"""

import math
import hashlib
from mpmath import csc
import numpy as np
import pywt

MARK_SIZE = 1024
BLOCK_SIZE = 4
ALPHA = 2
WAVELET = "haar"


def bytes_to_bits(b: bytes):
    bits = []
    for byte in b:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits


def build_mark_from_id(robust_id: str, mark_size: int = MARK_SIZE) -> np.ndarray:
    """
    Deterministically produce a (sqrt(mark_size), sqrt(mark_size)) binary watermark
    from a string ID, using SHA-256 and repetition.
    """
    side = int(math.isqrt(mark_size))
    if side * side != mark_size:
        raise ValueError("mark_size must be a perfect square (e.g., 1024 -> 32x32).")

    hashed = hashlib.sha256(robust_id.encode("utf-8")).digest()  # 256 bits
    bits = bytes_to_bits(hashed)  # length 256
    reps = int(math.ceil(mark_size / len(bits)))
    bits = (bits * reps)[:mark_size]
    return np.array(bits, dtype=np.float64).reshape(side, side)


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
    Y = np.matmul(np.matmul(V, imm), V.T)
    return Y


def iapdcbt(Y):
    M, N = Y.shape
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

    Vinv = np.linalg.inv(V)
    X = Vinv @ Y @ Vinv.T
    return X


def hide_mark(image_after_apdcbt, mark, alpha=ALPHA):
    dimx, dimy = image_after_apdcbt.shape

    Y_vec = np.reshape(image_after_apdcbt, (dimx * dimy,))
    Y_sgn = np.sign(Y_vec)
    Y_mod = np.abs(Y_vec)
    Y_index = np.argsort(-Y_mod, axis=None)

    mark = np.asarray(mark).reshape((-1,))
    wm_len = int(mark.size)

    if (dimx * dimy) - 1 < wm_len:
        raise ValueError("Image (after transform) is too small to embed the requested watermark length.")

    Yw_mod = Y_mod.copy()
    for idx, loc in enumerate(Y_index[1: wm_len + 1]):
        Yw_mod[loc] = Y_mod[loc] + alpha * mark[idx]

    Y_new_vec = np.multiply(Yw_mod, Y_sgn)
    Y_new = np.reshape(Y_new_vec, (dimx, dimy))
    return Y_new


def get_coefficient_matrix(image_block):
    nb_y, nb_x = image_block.shape[0], image_block.shape[1]
    for i in range(nb_y):
        for j in range(nb_x):
            image_block[i][j] = apdcbt(image_block[i][j])
    return image_block


def embedd_into_sub_band(sub_band, mark, block_size=BLOCK_SIZE):
    image_block = quantize(sub_band, block_size)
    coefficient_matrices = get_coefficient_matrix(image_block)

    nb_y, nb_x = coefficient_matrices.shape[0], coefficient_matrices.shape[1]
    coefficient_matrices = np.reshape(coefficient_matrices, (nb_y * block_size, nb_x * block_size))

    embedded = hide_mark(coefficient_matrices, mark, alpha=ALPHA)
    embedded = np.reshape(embedded, (nb_y, nb_x, block_size, block_size))

    watermarked_image_block = np.zeros((nb_y, nb_x, block_size, block_size))
    for i in range(nb_y):
        for j in range(nb_x):
            watermarked_image_block[i][j] = iapdcbt(embedded[i][j])

    sub_band_w = np.reshape(watermarked_image_block, (nb_y * block_size, nb_x * block_size))
    return sub_band_w


def embedding(image, mark):
    """
    Original signature preserved: embedding(image, mark) -> watermarked (uint8).
    Expects image as 2D grayscale array.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("embedding() expects a 2D grayscale image array.")

    # Ensure divisibility for 2-level DWT + block processing:
    # - 2-level DWT: divisible by 4
    # - block ops: divisible by 4 -> safest: 16
    image = _crop_to_multiple(image, 16)

    LL, (LH, HL, HH) = pywt.dwt2(image, WAVELET)
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL, WAVELET)

    mark = np.asarray(mark).reshape((-1,))
    LH2_w = embedd_into_sub_band(LH2, mark, block_size=BLOCK_SIZE)
    HL2_w = embedd_into_sub_band(HL2, mark, block_size=BLOCK_SIZE)

    LL_w = pywt.idwt2((LL2, (LH2_w, HL2_w, HH2)), WAVELET)
    watermarked = pywt.idwt2((LL_w, (LH, HL, HH)), WAVELET)

    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    return watermarked
