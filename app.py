import io
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from PIL import Image

from src import attacks
from src.fragile import fragile_lsb
from src.robust import embedding as robust_embedding
from src.robust import detection as robust_detection


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(
    page_title="Dual Watermarking (Robust ID + Fragile Integrity)",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Constants
# ----------------------------
FRAGILE_BLOCK_SIZE = 16
DEFAULT_SECRET_KEY = "my-secret-key"

ROBUST_BLOCK_SIZE = robust_embedding.BLOCK_SIZE
ROBUST_ALPHA = robust_embedding.ALPHA
ROBUST_WAVELET = robust_embedding.WAVELET
ROBUST_MARK_SIZE = robust_embedding.MARK_SIZE
ROBUST_THRESHOLD = robust_detection.THRESHOLD


# ----------------------------
# Image helpers
# ----------------------------
def crop_rgb_to_multiple(img: Image.Image, multiple: int = 16) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    w2 = (w // multiple) * multiple
    h2 = (h // multiple) * multiple
    return img.crop((0, 0, w2, h2))


def rgb_to_ycbcr_channels(img_rgb: Image.Image):
    ycbcr = img_rgb.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    return y, cb, cr


def ycbcr_to_rgb(y: Image.Image, cb: Image.Image, cr: Image.Image) -> Image.Image:
    ycbcr = Image.merge("YCbCr", (y, cb, cr))
    return ycbcr.convert("RGB")


def robust_embed_on_luma(img_rgb: Image.Image, robust_id: str):
    """
    Apply robust watermark on luminance (Y).
    Returns (rgb_out, meta_npz_bytes, meta_json_bytes).
    """
    img_rgb = crop_rgb_to_multiple(img_rgb, 16)

    y, cb, cr = rgb_to_ycbcr_channels(img_rgb)
    y_np = np.asarray(y, dtype=np.uint8)

    mark2d = robust_embedding.build_mark_from_id(robust_id, ROBUST_MARK_SIZE)
    mark_flat = mark2d.reshape((-1,))

    reference = robust_detection.compute_reference(
        y_np,
        block_size=ROBUST_BLOCK_SIZE,
        wavelet=ROBUST_WAVELET,
        mark_size=ROBUST_MARK_SIZE,
    )

    y_wm = robust_embedding.embedding(y_np, mark_flat)
    y_wm_img = Image.fromarray(y_wm, mode="L")
    rgb_out = ycbcr_to_rgb(y_wm_img, cb, cr)

    id_sha256 = hashlib.sha256(robust_id.encode("utf-8")).hexdigest()

    meta_core = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "robust_id": robust_id,
        "robust_id_sha256": id_sha256,
        "params": {
            "wavelet": ROBUST_WAVELET,
            "block_size": int(ROBUST_BLOCK_SIZE),
            "alpha": float(ROBUST_ALPHA),
            "threshold": float(ROBUST_THRESHOLD),
            "mark_size": int(ROBUST_MARK_SIZE),
        },
        "image_shape_hw": [int(reference["shape_hw"][0]), int(reference["shape_hw"][1])],
        "note": "Arrays (locations + reference magnitudes + mark) are stored in the companion .npz file.",
    }

    npz_buf = io.BytesIO()
    np.savez_compressed(
        npz_buf,
        meta_json=json.dumps(meta_core).encode("utf-8"),
        loc_LH2=reference["loc_LH2"],
        ref_LH2=reference["ref_LH2"],
        loc_HL2=reference["loc_HL2"],
        ref_HL2=reference["ref_HL2"],
        shape_hw=reference["shape_hw"],
        block_size=np.array([reference["block_size"]], dtype=np.int32),
        mark_size=np.array([reference["mark_size"]], dtype=np.int32),
        alpha=np.array([ROBUST_ALPHA], dtype=np.float32),
        threshold=np.array([ROBUST_THRESHOLD], dtype=np.float32),
        wavelet=np.array([ROBUST_WAVELET], dtype=object),
        mark=mark2d.astype(np.uint8),
    )
    npz_bytes = npz_buf.getvalue()

    json_bytes = json.dumps(meta_core, ensure_ascii=False, indent=2).encode("utf-8")
    return rgb_out, npz_bytes, json_bytes


def robust_verify_from_npz(img_rgb: Image.Image, npz_bytes: bytes):
    """
    Verify robust watermark using the .npz package (semi-blind).
    """
    data = np.load(io.BytesIO(npz_bytes), allow_pickle=True)

    meta_json = json.loads(data["meta_json"].tobytes().decode("utf-8"))
    loc_LH2 = data["loc_LH2"]
    ref_LH2 = data["ref_LH2"]
    loc_HL2 = data["loc_HL2"]
    ref_HL2 = data["ref_HL2"]
    shape_hw = data["shape_hw"]
    block_size = int(data["block_size"][0])
    mark_size = int(data["mark_size"][0])
    alpha = float(data["alpha"][0])
    threshold = float(data["threshold"][0])
    wavelet = str(data["wavelet"][0])
    expected_mark = data["mark"].astype(np.float64)

    img_rgb = crop_rgb_to_multiple(img_rgb, 16)
    y, _, _ = rgb_to_ycbcr_channels(img_rgb)
    y_np = np.asarray(y, dtype=np.uint8)

    reference = {
        "loc_LH2": loc_LH2,
        "ref_LH2": ref_LH2,
        "loc_HL2": loc_HL2,
        "ref_HL2": ref_HL2,
        "shape_hw": shape_hw,
        "block_size": block_size,
        "wavelet": wavelet,
        "mark_size": mark_size,
    }

    w1, w2 = robust_detection.extract_watermark_from_reference(y_np, reference, alpha=alpha)
    extracted_mark = robust_detection.merge_watermarks(w1, w2).astype(np.uint8)

    found = robust_detection.check_wm(
        expected_mark.reshape((-1,)),
        extracted_mark.reshape((-1,)),
        threshold=threshold,
    )
    sim = robust_detection.similarity(expected_mark.reshape((-1,)), extracted_mark.reshape((-1,)))

    return {
        "meta": meta_json,
        "found": bool(found),
        "similarity": float(sim),
        "expected_mark": expected_mark.astype(np.uint8),
        "extracted_mark": extracted_mark,
    }


def apply_attack_to_rgb_luma(img_rgb: Image.Image, attack_name: str, **params) -> Image.Image:
    """
    Apply a selected attack to the luminance channel only, keeping chroma unchanged.
    """
    img_rgb = crop_rgb_to_multiple(img_rgb, 16)
    y, cb, cr = rgb_to_ycbcr_channels(img_rgb)
    y_np = np.asarray(y, dtype=np.uint8)

    y_attacked = attacks.apply_attack(y_np, attack_name, **params)
    y_attacked_img = Image.fromarray(y_attacked, mode="L")
    return ycbcr_to_rgb(y_attacked_img, cb, cr)


# ----------------------------
# UI
# ----------------------------
st.sidebar.title("🔒 Dual Watermarking")
st.sidebar.markdown("---")

secret_key = st.sidebar.text_input(
    "Fragile Secret Key",
    value=DEFAULT_SECRET_KEY,
    help="Used for fragile LSB block-hash watermarking (integrity).",
)

robust_id = st.sidebar.text_input(
    "Robust ID (Assign an ID)",
    value="ID-0001",
    help="This ID is deterministically converted to a robust wavelet watermark.",
)

st.sidebar.markdown("### Robust Parameters (wavelet)")
st.sidebar.caption("Keep these aligned with src/robust/*.py modules.")
st.sidebar.write(f"ALPHA (fixed): **{ROBUST_ALPHA}**")
st.sidebar.write(f"Threshold (used at verification): **{ROBUST_THRESHOLD}**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["1) Embed (Robust ID + Fragile Integrity)", "2) Attack (Only attacks.py)", "3) Verify (Robust + Fragile)"],
)


# ----------------------------
# Page: Embed
# ----------------------------
if page.startswith("1)"):
    st.title("1) Embed: Robust ID + Fragile Integrity")
    st.markdown(
        """
This page performs **two-stage watermarking**:

1. **Robust (Wavelet) watermark** embeds an ID-derived mark (attack-resistant).
2. **Fragile (LSB hash) watermark** embeds a block-hash for integrity/tamper detection.
"""
    )

    uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        original = Image.open(uploaded)
        original_proc = crop_rgb_to_multiple(original, 16)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original (preprocessed)")
            st.image(original_proc, use_container_width=True)

        with st.spinner("Embedding robust ID watermark..."):
            robust_rgb, meta_npz_bytes, meta_json_bytes = robust_embed_on_luma(original_proc, robust_id)

        with st.spinner("Embedding fragile integrity watermark..."):
            final_img = fragile_lsb.embed_watermark(robust_rgb, secret_key, FRAGILE_BLOCK_SIZE)

        with col2:
            st.subheader("Final Watermarked Image")
            st.image(final_img, use_container_width=True)

        st.markdown("### Quality Metrics (Original vs Final)")
        psnr_value = fragile_lsb.calculate_psnr(original_proc, final_img)
        ssim_value = fragile_lsb.calculate_ssim(original_proc, final_img)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("PSNR (dB)", "∞" if psnr_value == float("inf") else f"{psnr_value:.2f}")
        with c2:
            st.metric("SSIM", f"{ssim_value:.4f}")

        st.markdown("### Downloads")
        st.download_button(
            "Download Final Watermarked Image (PNG)",
            data=fragile_lsb.pil_image_to_png_bytes(final_img),
            file_name="watermarked_dual.png",
            mime="image/png",
        )
        st.download_button(
            "Download Robust Metadata (.npz)",
            data=meta_npz_bytes,
            file_name=f"robust_meta_{robust_id}.npz",
            mime="application/octet-stream",
            help="Contains reference arrays + expected watermark mark for robust verification.",
        )
        st.download_button(
            "Download Robust Metadata (.json)",
            data=meta_json_bytes,
            file_name=f"robust_meta_{robust_id}.json",
            mime="application/json",
            help="Human-readable metadata (arrays are stored in the .npz).",
        )

        with st.expander("What is inside the robust .npz?"):
            st.markdown(
                """
- `loc_LH2`, `ref_LH2`: top coefficient locations + reference magnitudes (cover)
- `loc_HL2`, `ref_HL2`: same for the vertical sub-band
- `mark`: expected 32×32 binary watermark derived from your ID
- `alpha`, `threshold`, `block_size`, `wavelet`, `shape_hw`: parameters needed for verification
"""
            )


# ----------------------------
# Page: Attack
# ----------------------------
elif page.startswith("2)"):
    st.title("2) Attack Laboratory (Only src/attacks.py)")
    st.markdown(
        """
This page applies **only** the attacks implemented in `src/attacks.py`.
Attacks are applied to the **luminance (Y)** channel to stress-test the robust watermark,
and the result is saved as a new image for verification.
"""
    )

    uploaded = st.file_uploader("Upload a watermarked image to attack", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded)
        img = crop_rgb_to_multiple(img, 16)

        attack_name = st.selectbox("Attack type", list(attacks.ATTACK_REGISTRY.keys()))
        default_params = attacks.ATTACK_REGISTRY[attack_name]["params"]

        st.markdown("### Attack parameters")
        params = {}
        if attack_name == "AWGN":
            params["std"] = st.slider("std", 1.0, 50.0, float(default_params["std"]), 0.5)
            params["seed"] = st.number_input("seed", value=int(default_params["seed"]), step=1)
        elif attack_name == "Blur":
            params["sigma"] = st.slider("sigma", 0.1, 5.0, float(default_params["sigma"]), 0.1)
        elif attack_name == "Sharpening":
            params["sigma"] = st.slider("sigma", 0.1, 5.0, float(default_params["sigma"]), 0.1)
            params["alpha"] = st.slider("alpha", 0.1, 5.0, float(default_params["alpha"]), 0.1)
        elif attack_name == "Median":
            params["kernel_size"] = st.selectbox("kernel_size", [3, 5, 7], index=0)
        elif attack_name == "Resizing":
            params["scale"] = st.selectbox("scale", [0.25, 0.5, 0.75], index=1)
        elif attack_name == "JPEG":
            params["QF"] = st.slider("QF (quality)", 10, 100, int(default_params["QF"]), 1)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(img, use_container_width=True)

        attacked_img = None
        if st.button("Apply attack"):
            with st.spinner("Applying attack..."):
                attacked_img = apply_attack_to_rgb_luma(img, attack_name, **params)

        if attacked_img is not None:
            with col2:
                st.subheader("Attacked")
                st.image(attacked_img, use_container_width=True)

            y0, _, _ = rgb_to_ycbcr_channels(img)
            y1, _, _ = rgb_to_ycbcr_channels(attacked_img)
            wpsnr_val = robust_detection.wpsnr(np.asarray(y0, dtype=np.uint8), np.asarray(y1, dtype=np.uint8))
            st.metric("wPSNR (Y channel)", f"{wpsnr_val:.2f} dB" if wpsnr_val < 1e6 else "∞")

            st.download_button(
                "Download Attacked Image (PNG)",
                data=fragile_lsb.pil_image_to_png_bytes(attacked_img),
                file_name=f"attacked_{attack_name}.png",
                mime="image/png",
            )


# ----------------------------
# Page: Verify
# ----------------------------
else:
    st.title("3) Verify: Robust + Fragile")
    st.markdown(
        """
Verification has **two independent outputs**:

- **Robust**: checks whether the embedded ID-watermark is still detectable (attack-resilient).
- **Fragile**: checks whether image integrity is preserved (tamper detection).
"""
    )

    col_left, col_right = st.columns(2)

    with col_left:
        suspect_file = st.file_uploader("Upload suspect image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        meta_file = st.file_uploader("Upload robust metadata (.npz)", type=["npz"])

    with col_right:
        st.markdown("### Notes")
        st.markdown(
            """
- Robust verification uses the `.npz` metadata created at embedding time (semi-blind reference).
- Fragile verification uses the **Fragile Secret Key** from the sidebar.
"""
        )

    if suspect_file is not None:
        suspect_img = Image.open(suspect_file)
        suspect_img = crop_rgb_to_multiple(suspect_img, 16)

        st.subheader("Suspect Image")
        st.image(suspect_img, use_container_width=True)

        with st.spinner("Verifying fragile integrity watermark..."):
            tamper_map, tampered_blocks = fragile_lsb.verify_watermark(suspect_img, secret_key, FRAGILE_BLOCK_SIZE)

        st.markdown("## Fragile Integrity Result")
        if tampered_blocks == 0:
            st.success("SECURE — No tampering detected by fragile watermark.")
        else:
            st.error(f"TAMPERED — {tampered_blocks} blocks flagged by fragile watermark.")

        overlay = fragile_lsb.create_overlay_image(
            fragile_lsb.preprocess_image(suspect_img, FRAGILE_BLOCK_SIZE),
            tamper_map,
            alpha=0.4,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(fragile_lsb.preprocess_image(suspect_img, FRAGILE_BLOCK_SIZE), caption="Processed (fragile)", use_container_width=True)
        with c2:
            st.image(tamper_map, caption="Tamper map (red=changed)", use_container_width=True)
        with c3:
            st.image(overlay, caption="Overlay", use_container_width=True)

        if meta_file is not None:
            meta_npz_bytes = meta_file.getvalue()
            with st.spinner("Verifying robust ID watermark..."):
                robust_res = robust_verify_from_npz(suspect_img, meta_npz_bytes)

            st.markdown("## Robust ID Result")
            meta = robust_res["meta"]
            st.write(f"**Expected Robust ID:** `{meta['robust_id']}`")
            st.write(f"**Similarity:** {robust_res['similarity']:.3f}")
            if robust_res["found"]:
                st.success("ROBUST WATERMARK FOUND — ID watermark is still detectable.")
            else:
                st.error("ROBUST WATERMARK LOST — ID watermark could not be reliably detected.")

            c1, c2 = st.columns(2)
            with c1:
                st.image(robust_res["expected_mark"], caption="Expected mark (from ID)", use_container_width=True)
            with c2:
                st.image(robust_res["extracted_mark"], caption="Extracted mark (from suspect)", use_container_width=True)

        else:
            st.info("Upload the robust metadata (.npz) to run robust verification.")
