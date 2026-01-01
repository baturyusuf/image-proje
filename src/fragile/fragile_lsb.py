"""
Fragile watermarking (LSB block hashing) extracted from your Streamlit logic.

Pure-Python (no Streamlit), so app.py remains UI-only.
"""

import io
import math
import random
import hashlib
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def bytes_to_bits(data_bytes: bytes):
    bits = []
    for byte in data_bytes:
        for i in range(8):
            bit = (byte >> (7 - i)) & 1
            bits.append(bit)
    return bits


def bits_to_bytes(bits_list):
    if len(bits_list) % 8 != 0:
        raise ValueError("Bit list length must be multiple of 8.")

    data_bytes = bytearray()
    for i in range(0, len(bits_list), 8):
        byte_bits = bits_list[i:i + 8]
        byte_val = 0
        for bit in byte_bits:
            byte_val = (byte_val << 1) | int(bit)
        data_bytes.append(byte_val)
    return bytes(data_bytes)


def create_block_mapping(num_blocks: int, key: str):
    indices = list(range(num_blocks))
    rng = random.Random(key)
    rng.shuffle(indices)
    return indices


def crop_to_multiple_rgb(pil_image: Image.Image, multiple: int):
    img = pil_image.convert("RGB")
    w, h = img.size
    w2 = (w // multiple) * multiple
    h2 = (h // multiple) * multiple
    return img.crop((0, 0, w2, h2))


def preprocess_image(pil_image: Image.Image, block_size: int):
    """
    Keep RGB (no grayscale conversion) and crop to a multiple of block_size.
    """
    return crop_to_multiple_rgb(pil_image, block_size)


def calculate_psnr(original: Image.Image, watermarked: Image.Image):
    original_array = np.array(original, dtype=np.float64)
    watermarked_array = np.array(watermarked, dtype=np.float64)

    mse = np.mean((original_array - watermarked_array) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    return 20.0 * math.log10(max_pixel / math.sqrt(mse))


def calculate_ssim(original: Image.Image, watermarked: Image.Image):
    original_array = np.array(original, dtype=np.float64)
    watermarked_array = np.array(watermarked, dtype=np.float64)

    if original_array.ndim == 3:
        original_gray = np.dot(original_array[..., :3], [0.299, 0.587, 0.114])
        watermarked_gray = np.dot(watermarked_array[..., :3], [0.299, 0.587, 0.114])
    else:
        original_gray = original_array
        watermarked_gray = watermarked_array

    original_norm = original_gray / 255.0
    watermarked_norm = watermarked_gray / 255.0

    ssim_value, _ = ssim(original_norm, watermarked_norm, data_range=1.0, full=True)
    return float(ssim_value)


def embed_watermark(original_pil_image: Image.Image, key: str, block_size: int):
    """
    Embed SHA-256(block) into BLUE channel LSBs.
    """
    img = preprocess_image(original_pil_image, block_size)
    width, height = img.size
    img_array = np.array(img, dtype=np.uint8)

    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    num_blocks_total = num_blocks_x * num_blocks_y

    hash_list = []
    for i in range(num_blocks_total):
        by = (i // num_blocks_x) * block_size
        bx = (i % num_blocks_x) * block_size

        block_data_blue = img_array[by:by + block_size, bx:bx + block_size, 2]
        msb_data = (block_data_blue & 0xFE).astype(np.uint8)

        msb_bytes = bytes(msb_data.flatten())
        hash_input = msb_bytes + key.encode("utf-8")
        block_hash = hashlib.sha256(hash_input).digest()
        hash_list.append(block_hash)

    mapping = create_block_mapping(num_blocks_total, key)
    watermarked_array = np.copy(img_array)

    for i in range(num_blocks_total):
        hash_bits = bytes_to_bits(hash_list[i])
        j = mapping[i]

        by = (j // num_blocks_x) * block_size
        bx = (j % num_blocks_x) * block_size

        target_block_blue = watermarked_array[by:by + block_size, bx:bx + block_size, 2]
        target_pixels_flat = target_block_blue.flatten()

        for p_idx in range(block_size * block_size):
            pixel_val = target_pixels_flat[p_idx]
            bit_to_embed = hash_bits[p_idx]
            target_pixels_flat[p_idx] = (pixel_val & 0xFE) | bit_to_embed

        watermarked_array[by:by + block_size, bx:bx + block_size, 2] = target_pixels_flat.reshape((block_size, block_size))

    return Image.fromarray(watermarked_array, "RGB")


def verify_watermark(image_to_check_pil: Image.Image, key: str, block_size: int):
    """
    Verify fragile watermark and return (tamper_map_img, tampered_blocks).
    """
    img = preprocess_image(image_to_check_pil, block_size)
    width, height = img.size
    img_array = np.array(img, dtype=np.uint8)

    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    num_blocks_total = num_blocks_x * num_blocks_y

    tamper_map_array = np.zeros((height, width, 3), dtype=np.uint8)

    mapping = create_block_mapping(num_blocks_total, key)
    tampered_blocks = 0

    for i in range(num_blocks_total):
        by_i = (i // num_blocks_x) * block_size
        bx_i = (i % num_blocks_x) * block_size

        j = mapping[i]
        by_j = (j // num_blocks_x) * block_size
        bx_j = (j % num_blocks_x) * block_size

        target_block_blue = img_array[by_j:by_j + block_size, bx_j:bx_j + block_size, 2]
        target_pixels_flat = target_block_blue.flatten()

        embedded_bits = [(int(v) & 1) for v in target_pixels_flat[: block_size * block_size]]
        H_i_embedded = bits_to_bytes(embedded_bits)

        block_data_i_blue = img_array[by_i:by_i + block_size, bx_i:bx_i + block_size, 2]
        msb_data_i = (block_data_i_blue & 0xFE).astype(np.uint8)

        msb_bytes_i = bytes(msb_data_i.flatten())
        hash_input_i = msb_bytes_i + key.encode("utf-8")
        H_i_current = hashlib.sha256(hash_input_i).digest()

        if H_i_embedded != H_i_current:
            tampered_blocks += 1
            tamper_map_array[by_i:by_i + block_size, bx_i:bx_i + block_size, 0] = 255

    return Image.fromarray(tamper_map_array, "RGB"), int(tampered_blocks)


def create_overlay_image(original_img: Image.Image, tamper_map: Image.Image, alpha: float = 0.4):
    original_rgb = original_img.convert("RGB")
    return Image.blend(original_rgb, tamper_map.convert("RGB"), alpha=float(alpha))


def pil_image_to_png_bytes(pil_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
