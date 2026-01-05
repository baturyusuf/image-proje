# Dual Watermarking Web App (Robust ID + Fragile Integrity)

A **Streamlit** web application that implements **dual watermarking** on images:

* **Robust watermark (Wavelet-based ID)**: embeds an **ID-derived** watermark designed to remain detectable under common signal-processing attacks (noise, blur, resizing, JPEG, etc.).
* **Fragile watermark (LSB block-hash)**: embeds **block-level integrity hashes** that break under modification to enable **tamper localization**.

This combination supports two complementary goals:

1. **Ownership / identification** under benign-to-moderate distortions (robust).
2. **Integrity verification** and tamper visualization (fragile).

---

## Key Features

### 1) Embed (Robust + Fragile)

* Upload any PNG/JPG image
* Robust watermark is embedded into the **luminance (Y)** channel in the wavelet domain
* Fragile watermark is embedded into the **blue-channel LSBs** as SHA-256 block hashes
* Quality metrics reported:

  * **PSNR**
  * **SSIM**
* Download outputs:

  * Final watermarked image (`.png`)
  * Robust metadata (`.npz` + `.json`) for semi-blind verification

### 2) Attack Laboratory

Applies attacks **only from `src/attacks.py`**, in-memory and reproducible:

* AWGN (Additive White Gaussian Noise)
* Blur (Gaussian)
* Sharpening
* Median filtering
* Resizing (downscale + upscale)
* JPEG compression (in-memory)

Attacks are applied to the **Y channel** to stress-test robust detection while keeping chroma channels unchanged.

### 3) Verify (Robust + Fragile)

* **Fragile verification**:

  * Detects tampered blocks
  * Produces a **tamper map** (red = changed)
  * Produces an **overlay** visualization
* **Robust verification** (semi-blind):

  * Requires the `.npz` metadata generated at embed time
  * Extracts watermark from suspect image and checks:

    * `found` (boolean)
    * `similarity` score
  * Visualizes expected vs extracted watermark patterns

---

## Project Structure

```text
baturyusuf-image-proje/
├── app.py
├── README.md
├── requirements.txt
├── assets/
│   └── csf.csv
└── src/
    ├── attacks.py
    ├── fragile/
    │   └── fragile_lsb.py
    └── robust/
        ├── embedding.py
        └── detection.py
```

**Module responsibilities**

* `app.py`: Streamlit UI and orchestration (embed / attack / verify pages)
* `src/fragile/fragile_lsb.py`: fragile LSB block-hash watermarking + PSNR/SSIM + visual overlays
* `src/robust/embedding.py`: robust ID embedding using DWT + APDCBT-based coefficient selection
* `src/robust/detection.py`: robust extraction/verification, semi-blind reference flow (`.npz`), and wPSNR
* `src/attacks.py`: attack suite for robustness testing

---

## Installation

### 1) Create and activate a virtual environment (recommended)

**Windows (PowerShell)**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

Then use the sidebar navigation:

1. **Embed (Robust ID + Fragile Integrity)**
2. **Attack (Only attacks.py)**
3. **Verify (Robust + Fragile)**

---

## How It Works

## Robust Watermark (Wavelet ID)

* The robust watermark is embedded into the **Y (luminance)** channel.
* An input string (e.g., `ID-0001`) is hashed and expanded into a **binary watermark matrix** (default: `32×32`).
* Embedding occurs in the **2-level DWT** sub-bands (LH2 and HL2), using coefficient modulation controlled by:

  * `ALPHA`
  * `BLOCK_SIZE`
  * `WAVELET`
  * `MARK_SIZE`

### Semi-blind verification (metadata-driven)

Instead of requiring the full original image during verification, the system stores a **reference package**:

* locations of selected coefficients
* reference magnitudes from the cover image
* expected watermark mark

These are stored in the downloadable `.npz`, enabling verification later without the original image.

---

## Fragile Watermark (LSB Block Hashing)

* The image is divided into fixed blocks (default: `16×16`).
* For each block, a **SHA-256 hash** is computed from:

  * the block’s **MSB content** (LSB removed)
  * the user’s **secret key**
* The resulting 256 hash bits are embedded into the **LSBs** of the block pixels (blue channel).
* During verification, the embedded hash and recomputed hash are compared:

  * mismatch ⇒ block is flagged as **tampered**
  * output ⇒ **tamper map** and count of tampered blocks

---

## Robust Metadata Files

When embedding, you can download:

### `robust_meta_<ID>.json`

Human-readable metadata:

* creation timestamp
* robust ID and hash
* parameters (alpha, threshold, wavelet, etc.)
* image shape

### `robust_meta_<ID>.npz`

Machine-readable reference package (used by verification):

* coefficient locations: `loc_LH2`, `loc_HL2`
* cover reference magnitudes: `ref_LH2`, `ref_HL2`
* expected mark matrix: `mark`
* shape and parameters

---

## Notes on Quality Metrics

* **PSNR / SSIM** are reported for *Original vs Final (Robust+Fragile embedded)*.
* **wPSNR (Y channel)** is reported in the attack lab to quantify perceptual degradation after attacks.

---

## Configuration

Key constants used by the app (see `app.py`):

* Fragile:

  * `FRAGILE_BLOCK_SIZE = 16`
  * `DEFAULT_SECRET_KEY = "my-secret-key"` (change this)

* Robust (from `src/robust/*.py`):

  * `BLOCK_SIZE`
  * `ALPHA`
  * `WAVELET`
  * `MARK_SIZE`
  * `THRESHOLD`

Recommendation:

* Keep robust parameters consistent between embed and verify.
* Use a strong, unique fragile secret key for meaningful integrity verification.

---

## Example Workflow

1. **Embed**

   * Upload image
   * Enter `Robust ID`
   * Enter `Fragile Secret Key`
   * Download:

     * `watermarked_dual.png`
     * `robust_meta_<ID>.npz`

2. **Attack**

   * Upload `watermarked_dual.png`
   * Apply an attack (e.g., JPEG QF=50)
   * Download attacked image

3. **Verify**

   * Upload attacked image
   * Upload `robust_meta_<ID>.npz`
   * Inspect:

     * Fragile tamper map (should often flag changes under attacks)
     * Robust similarity / detection status (should remain detectable under moderate attacks)

---

## Security and Practical Considerations

* The **fragile watermark** is intended for **tamper detection**, so it is expected to break under many transformations (including compression and resizing).
* The **robust watermark** is intended for **ID persistence** under common distortions, but extreme transformations can still cause loss.
* This repository demonstrates a practical dual-watermark workflow rather than claiming cryptographic security guarantees.

---

## Dependencies

* `streamlit` – UI framework
* `numpy`, `pillow` – image processing utilities
* `pywavelets` – wavelet transforms (robust watermark)
* `scipy`, `scikit-image` – filtering, resizing, SSIM
* `mpmath` – APDCBT helper math

