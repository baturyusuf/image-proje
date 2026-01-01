import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import hashlib
import random
import io
import math
from skimage.metrics import structural_similarity as ssim

# Page configuration
st.set_page_config(
    page_title="Invisible Fragile Watermarking",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BLOCK_SIZE = 16
DEFAULT_SECRET_KEY = "my-secret-key"

# Core watermarking functions
def bytes_to_bits(data_bytes):
    """Convert bytes to list of bits (0/1)."""
    bits = []
    for byte in data_bytes:
        for i in range(8):
            bit = (byte >> (7 - i)) & 1
            bits.append(bit)
    return bits

def bits_to_bytes(bits_list):
    """Convert list of bits to bytes."""
    if len(bits_list) % 8 != 0:
        raise ValueError("Bit list length must be multiple of 8.")

    data_bytes = bytearray()
    for i in range(0, len(bits_list), 8):
        byte_bits = bits_list[i:i+8]
        byte_val = 0
        for bit in byte_bits:
            byte_val = (byte_val << 1) | bit
        data_bytes.append(byte_val)
    return bytes(data_bytes)

def create_block_mapping(num_blocks, key):
    """Create deterministic block mapping based on secret key."""
    indices = list(range(num_blocks))
    rng = random.Random(key)
    rng.shuffle(indices)
    return indices

def preprocess_image(pil_image, block_size):
    """Resize RGB image to be divisible by block_size without changing color mode."""
    # Keep RGB mode for maximum visual fidelity
    if pil_image.mode != 'RGB':
        img = pil_image.convert('RGB')
    else:
        img = pil_image

    width, height = img.size
    width = (width // block_size) * block_size
    height = (height // block_size) * block_size
    img = img.crop((0, 0, width, height))
    return img

def calculate_psnr(original, watermarked):
    """Calculate Peak Signal-to-Noise Ratio."""
    original_array = np.array(original, dtype=np.float64)
    watermarked_array = np.array(watermarked, dtype=np.float64)

    mse = np.mean((original_array - watermarked_array) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ssim(original, watermarked):
    """Calculate Structural Similarity Index."""
    original_array = np.array(original, dtype=np.float64)
    watermarked_array = np.array(watermarked, dtype=np.float64)

    # Convert to grayscale for SSIM if RGB
    if len(original_array.shape) == 3:
        # Use luminance weights for RGB to grayscale conversion
        original_gray = np.dot(original_array[..., :3], [0.299, 0.587, 0.114])
        watermarked_gray = np.dot(watermarked_array[..., :3], [0.299, 0.587, 0.114])
    else:
        original_gray = original_array
        watermarked_gray = watermarked_array

    # Normalize to 0-1 range for SSIM
    original_norm = original_gray / 255.0
    watermarked_norm = watermarked_gray / 255.0

    ssim_value, _ = ssim(original_norm, watermarked_norm, data_range=1.0, full=True)
    return ssim_value

def embed_watermark(_original_pil_image, key, block_size):
    """Embed watermark into RGB image using Blue channel LSB for maximum visual fidelity."""
    img = preprocess_image(_original_pil_image, block_size)
    width, height = img.size
    img_array = np.array(img, dtype=np.uint8)  # Shape: (height, width, 3) for RGB

    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    num_blocks_total = num_blocks_x * num_blocks_y

    # Create hash list for all blocks using Blue channel MSB data
    hash_list = []
    for i in range(num_blocks_total):
        by = (i // num_blocks_x) * block_size
        bx = (i % num_blocks_x) * block_size
        # Use Blue channel (index 2) for hashing to maintain consistency
        block_data_blue = img_array[by:by+block_size, bx:bx+block_size, 2]  # Blue channel
        msb_data = (block_data_blue & 0xFE).astype(np.uint8)

        msb_bytes = bytes(msb_data.flatten())
        hash_input = msb_bytes + key.encode('utf-8')
        block_hash = hashlib.sha256(hash_input).digest()
        hash_list.append(block_hash)

    # Create block mapping
    mapping = create_block_mapping(num_blocks_total, key)

    # Embed hashes into Blue channel LSBs only (maximum visual fidelity)
    watermarked_array = np.copy(img_array)

    for i in range(num_blocks_total):
        hash_to_embed = hash_list[i]
        hash_bits = bytes_to_bits(hash_to_embed)
        j = mapping[i]

        by = (j // num_blocks_x) * block_size
        bx = (j % num_blocks_x) * block_size

        # Modify only Blue channel LSBs (index 2)
        target_block_blue = watermarked_array[by:by+block_size, bx:bx+block_size, 2]
        target_pixels_flat = target_block_blue.flatten()

        for p_idx in range(block_size * block_size):
            pixel_val = target_pixels_flat[p_idx]
            bit_to_embed = hash_bits[p_idx]
            # Strict LSB modification: maximum change of ±1
            new_pixel_val = (pixel_val & 0xFE) | bit_to_embed
            target_pixels_flat[p_idx] = new_pixel_val

        watermarked_array[by:by+block_size, bx:bx+block_size, 2] = target_pixels_flat.reshape((block_size, block_size))

    watermarked_img = Image.fromarray(watermarked_array, 'RGB')
    return watermarked_img

def verify_watermark(_image_to_check_pil, key, block_size):
    """Verify watermark and return tamper map and tampered block count."""
    img = preprocess_image(_image_to_check_pil, block_size)
    width, height = img.size
    img_array = np.array(img, dtype=np.uint8)  # Shape: (height, width, 3) for RGB

    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    num_blocks_total = num_blocks_x * num_blocks_y

    # Create tamper map (RGB)
    tamper_map_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Recreate mapping
    mapping = create_block_mapping(num_blocks_total, key)

    tampered_blocks = 0

    for i in range(num_blocks_total):
        by_i = (i // num_blocks_x) * block_size
        bx_i = (i % num_blocks_x) * block_size

        # Extract embedded hash from Blue channel LSBs
        j = mapping[i]
        by_j = (j // num_blocks_x) * block_size
        bx_j = (j % num_blocks_x) * block_size

        target_block_blue = img_array[by_j:by_j+block_size, bx_j:bx_j+block_size, 2]  # Blue channel
        target_pixels_flat = target_block_blue.flatten()

        embedded_bits = []
        for p_idx in range(block_size * block_size):
            lsb_bit = target_pixels_flat[p_idx] & 1
            embedded_bits.append(lsb_bit)

        H_i_embedded = bits_to_bytes(embedded_bits)

        # Calculate current hash from Blue channel MSB data
        block_data_i_blue = img_array[by_i:by_i+block_size, bx_i:bx_i+block_size, 2]  # Blue channel
        msb_data_i = (block_data_i_blue & 0xFE).astype(np.uint8)

        msb_bytes_i = bytes(msb_data_i.flatten())
        hash_input_i = msb_bytes_i + key.encode('utf-8')
        H_i_current = hashlib.sha256(hash_input_i).digest()

        # Compare
        if H_i_embedded != H_i_current:
            tampered_blocks += 1
            # Mark block as red in tamper map
            tamper_map_array[by_i:by_i+block_size, bx_i:bx_i+block_size, 0] = 255

    tamper_map_img = Image.fromarray(tamper_map_array, 'RGB')
    return tamper_map_img, tampered_blocks

def create_overlay_image(original_img, tamper_map, alpha=0.4):
    """Create overlay of original image with tamper map."""
    original_rgb = original_img.convert("RGB")
    overlay = Image.blend(original_rgb, tamper_map, alpha=alpha)
    return overlay

def add_random_noise(image, noise_percentage=0.1, amplitude=50):
    """Add random noise to image safely (avoid uint8 overflow)."""
    arr = np.array(image)

    # RGB
    if arr.ndim == 3:
        arr16 = arr.astype(np.int16)
        h, w, c = arr16.shape

        total_pixels = h * w
        pixels_to_modify = int(total_pixels * noise_percentage)
        if pixels_to_modify <= 0:
            return Image.fromarray(arr.astype(np.uint8), 'RGB')

        # Random pixel coordinates
        ys = np.random.randint(0, h, size=pixels_to_modify)
        xs = np.random.randint(0, w, size=pixels_to_modify)

        # Noise per channel
        noise = np.random.randint(-amplitude, amplitude + 1, size=(pixels_to_modify, c), dtype=np.int16)

        # Apply + clip
        arr16[ys, xs, :] = np.clip(arr16[ys, xs, :] + noise, 0, 255)

        return Image.fromarray(arr16.astype(np.uint8), 'RGB')

    # Grayscale
    else:
        arr16 = arr.astype(np.int16)
        h, w = arr16.shape

        total_pixels = h * w
        pixels_to_modify = int(total_pixels * noise_percentage)
        if pixels_to_modify <= 0:
            return Image.fromarray(arr.astype(np.uint8), 'L')

        ys = np.random.randint(0, h, size=pixels_to_modify)
        xs = np.random.randint(0, w, size=pixels_to_modify)

        noise = np.random.randint(-amplitude, amplitude + 1, size=pixels_to_modify, dtype=np.int16)

        arr16[ys, xs] = np.clip(arr16[ys, xs] + noise, 0, 255)

        return Image.fromarray(arr16.astype(np.uint8), 'L')

def add_text_overlay(image, text, font_size, text_color, x, y):
    """Add text overlay to image."""
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Create a copy to modify
    modified_image = image.copy()
    draw = ImageDraw.Draw(modified_image)

    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.load_default()
        except:
            font = None

    # Convert hex color to RGB tuple
    if text_color.startswith('#'):
        text_color = text_color[1:]
    r = int(text_color[0:2], 16)
    g = int(text_color[2:4], 16)
    b = int(text_color[4:6], 16)
    color_rgb = (r, g, b)

    # Draw text
    draw.text((x, y), text, fill=color_rgb, font=font)
    return modified_image

def add_geometric_shape(image, shape_type, shape_color, x, y, width, height):
    """Add geometric shapes to image."""
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Create a copy to modify
    modified_image = image.copy()
    draw = ImageDraw.Draw(modified_image)

    # Convert hex color to RGB tuple
    if shape_color.startswith('#'):
        shape_color = shape_color[1:]
    r = int(shape_color[0:2], 16)
    g = int(shape_color[2:4], 16)
    b = int(shape_color[4:6], 16)
    color_rgb = (r, g, b)

    # Draw shape based on type
    if shape_type == "Rectangle":
        draw.rectangle([x, y, x + width, y + height], fill=color_rgb)
    elif shape_type == "Circle":
        draw.ellipse([x, y, x + width, y + height], fill=color_rgb)
    elif shape_type == "Line":
        draw.line([x, y, x + width, y + height], fill=color_rgb, width=5)

    return modified_image

def crop_image(image, left, top, right, bottom):
    """Crop image to specified coordinates."""
    return image.crop((left, top, right, bottom))

# Main app
def main():
    # Sidebar
    st.sidebar.title("🔒 Invisible Fragile Watermarking")
    st.sidebar.markdown("---")

    # Global settings
    secret_key = st.sidebar.text_input(
        "Secret Key",
        value=DEFAULT_SECRET_KEY,
        help="Secret key used for watermarking and verification"
    )

    st.sidebar.markdown(f"**Block Size:** {BLOCK_SIZE}x{BLOCK_SIZE}")
    st.sidebar.markdown("---")

    # Page selection
    page = st.sidebar.radio(
        "Navigation",
        ["Mühürleme (Embed)", "Saldırı Laboratuvarı (Attack)", "Doğrulama (Verify)"],
        help="Select the operation you want to perform"
    )

    # Page content
    if page == "Mühürleme (Embed)":
        embed_page(secret_key, BLOCK_SIZE)
    elif page == "Saldırı Laboratuvarı (Attack)":
        attack_page(BLOCK_SIZE)
    elif page == "Doğrulama (Verify)":
        verify_page(secret_key, BLOCK_SIZE)

def embed_page(secret_key, block_size):
    st.title("🛡️ Mühürleme (Embed Watermark)")
    st.markdown("Upload an image to embed an invisible fragile watermark.")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        help="Upload the original image you want to watermark"
    )

    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_container_width=True)

        # Process image
        with st.spinner("Embedding watermark..."):
            watermarked_image = embed_watermark(original_image, secret_key, block_size)

            # Calculate quality metrics
            processed_original = preprocess_image(original_image, block_size)
            psnr_value = calculate_psnr(processed_original, watermarked_image)
            ssim_value = calculate_ssim(processed_original, watermarked_image)

        with col2:
            st.subheader("Watermarked Image")
            st.image(watermarked_image, use_column_width=True)

        # Metrics
        st.markdown("### Quality Metrics (Maximum Visual Fidelity)")

        # PSNR Display
        col_psnr, col_ssim = st.columns(2)
        with col_psnr:
            if psnr_value == float('inf'):
                st.success("**PSNR:** ∞ dB\n\n*Images are identical*")
            else:
                if psnr_value > 50:
                    st.success(f"**PSNR:** {psnr_value:.2f} dB ✅\n\n*Excellent quality preservation*")
                elif psnr_value > 30:
                    st.info(f"**PSNR:** {psnr_value:.2f} dB\n\n*Good quality*")
                else:
                    st.warning(f"**PSNR:** {psnr_value:.2f} dB\n\n*Quality degradation detected*")

        # SSIM Display
        with col_ssim:
            if ssim_value > 0.99:
                st.success(f"**SSIM:** {ssim_value:.4f} ✅\n\n*Perfect structural similarity*")
            elif ssim_value > 0.95:
                st.info(f"**SSIM:** {ssim_value:.4f}\n\n*High structural similarity*")
            else:
                st.warning(f"**SSIM:** {ssim_value:.4f}\n\n*Structural differences detected*")

        # Technical details about fidelity
        with st.expander("🔬 Fidelity Details"):
            st.markdown(f"""
            **Embedding Strategy:**
            - **RGB Mode**: No grayscale conversion for maximum color fidelity
            - **Blue Channel LSB**: Only Blue channel LSBs modified (±1 max change per pixel)
            - **Strict LSB Constraint**: Uses `val & 0xFE | bit` for minimal perturbation
            - **Block Size**: {block_size}x{block_size} pixels
            - **Hash Algorithm**: SHA-256 per block + secret key

            **Quality Targets:**
            - PSNR > 50 dB (excellent)
            - SSIM > 0.99 (near-perfect structural similarity)
            - Human eye cannot detect differences
            """)

        # Download button
        buf = io.BytesIO()
        watermarked_image.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="📥 Download Watermarked Image (PNG)",
            data=buf,
            file_name="watermarked_image.png",
            mime="image/png",
            help="Download the watermarked image as PNG (lossless format)"
        )

        # Technical details
        with st.expander("🔧 Technical Details"):
            st.markdown(f"""
            **Processing Summary:**
            - Block Size: {block_size}x{block_size}
            - Secret Key: `{secret_key[:8]}...`
            - PSNR: {psnr_value:.2f} dB

            **How it works:**
            1. Image is converted to grayscale and resized to be divisible by {block_size}
            2. Each {block_size}x{block_size} block gets a SHA-256 hash of its MSB data + secret key
            3. Hashes are embedded into LSBs of mapped blocks using the secret key
            4. Any tampering will change the hash, making the watermark fragile
            """)

def attack_page(block_size):
    st.title("🎯 Saldırı Laboratuvarı (Attack Laboratory)")
    st.markdown("Simulate tampering attacks on watermarked images.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Watermarked Image",
        type=["png", "jpg", "jpeg"],
        help="Upload a watermarked image to simulate attacks on"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        image = preprocess_image(image, block_size)


        st.markdown("### Attack Methods")

        attack_method = st.radio(
            "Choose attack method:",
            ["Random Noise"],  # Temporarily only Random Noise due to canvas compatibility issues
            horizontal=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        attacked_image = None

        if attack_method == "Random Noise":
            st.markdown("**Draw on the image** to simulate tampering (e.g., adding text or covering objects).")

            st.markdown("**Add random noise** to simulate image degradation or compression artifacts.")

            noise_percent = st.slider(
                "Noise percentage:",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                format="%d%%"
            )

            if st.button("Apply Random Noise"):
                with st.spinner("Adding noise..."):
                    attacked_image = add_random_noise(image, noise_percent / 100.0)
                    orig_arr = np.array(image)
                    atk_arr = np.array(attacked_image)

                    if orig_arr.ndim == 3:
                        changed_pixels = np.mean(np.any(orig_arr != atk_arr, axis=2))
                    else:
                        changed_pixels = np.mean(orig_arr != atk_arr)

                    st.metric("Changed pixel ratio", f"{changed_pixels * 100:.2f}%")

        if attacked_image is not None:
            with col2:
                st.subheader("Attacked (Random Noise)")
                st.image(attacked_image, use_container_width=True)

            # Download attacked image
            buf = io.BytesIO()
            attacked_image.save(buf, format="PNG")
            buf.seek(0)

            st.download_button(
                label="📥 Download Attacked Image",
                data=buf,
                file_name="attacked_random_noise.png",
                mime="image/png",
                help="Download the attacked image for verification testing"
            )

def verify_page(secret_key, block_size):
    st.title("🔍 Doğrulama (Verify Watermark)")
    st.markdown("Upload a suspect image to verify its integrity and detect tampering.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Suspect Image",
        type=["png", "jpg", "jpeg"],
        help="Upload the image you want to verify for tampering"
    )

    if uploaded_file is not None:
        suspect_image = Image.open(uploaded_file)
        file_bytes = uploaded_file.getvalue()
        st.caption(f"Uploaded file SHA256 (first 16): {hashlib.sha256(file_bytes).hexdigest()[:16]}")

        # Verify watermark
        with st.spinner("Verifying watermark..."):
            tamper_map, tampered_blocks = verify_watermark(suspect_image, secret_key, block_size)

        # Results
        st.markdown("### Verification Results")

        if tampered_blocks == 0:
            st.success("🛡️ **SECURE** - Image integrity verified!")
            st.markdown("No tampering detected. The image is authentic and unchanged.")
        else:
            st.error("⚠️ **TAMPERED** - Image has been modified!")
            st.markdown(f"**{tampered_blocks}** blocks show signs of tampering.")

        # Visualization
        processed_suspect = preprocess_image(suspect_image, block_size)
        overlay_image = create_overlay_image(processed_suspect, tamper_map, alpha=0.4)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Suspect Image")
            st.image(processed_suspect, use_column_width=True)

        with col2:
            st.subheader("Tamper Map")
            st.image(tamper_map, use_column_width=True)
            st.caption("Red areas indicate tampered blocks")

        with col3:
            st.subheader("Overlay")
            st.image(overlay_image, use_column_width=True)
            st.caption("Tamper map blended with suspect image")

        # Download tamper map
        buf = io.BytesIO()
        tamper_map.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="📥 Download Tamper Map",
            data=buf,
            file_name="tamper_map.png",
            mime="image/png",
            help="Download the tamper detection map"
        )

        # Technical details
        with st.expander("🔧 Technical Details"):
            processed_width, processed_height = processed_suspect.size
            total_blocks = (processed_width // block_size) * (processed_height // block_size)

            st.markdown(f"""
            **Verification Summary:**
            - Total blocks analyzed: {total_blocks}
            - Tampered blocks: {tampered_blocks}
            - Integrity: {((total_blocks - tampered_blocks) / total_blocks * 100):.1f}%
            - Secret Key: `{secret_key[:8]}...`

            **How verification works:**
            1. Recreate the block mapping using the secret key
            2. Extract embedded hashes from LSBs of mapped blocks
            3. Calculate current hashes from MSB data of each block
            4. Compare embedded vs. current hashes
            5. Mark blocks as tampered if hashes don't match
            """)

if __name__ == "__main__":
    main()
