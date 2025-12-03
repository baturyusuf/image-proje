# Invisible Fragile Watermarking Web App

A Streamlit-based web application for invisible fragile watermarking using LSB (Least Significant Bit) Block Hashing.

## Features

### 🛡️ Mühürleme (Embed)
- Upload images (PNG/JPG)
- Automatic preprocessing (grayscale, resize to 16x16 blocks)
- LSB watermark embedding with SHA-256 hashing
- PSNR quality measurement (>50dB target)
- PNG download (lossless format)

### 🎯 Saldırı Laboratuvarı (Attack Laboratory)
- Manual drawing attacks (simulate text addition/covering)
- Random noise attacks (simulate compression/degradation)
- Real-time canvas interaction
- Attack result download

### 🔍 Doğrulama (Verify)
- Tamper detection and visualization
- Red tamper map generation
- Overlay visualization
- Secure/Tampered status reporting

## Technical Details

- **Block Size**: 16x16 pixels (fixed)
- **Hash Algorithm**: SHA-256
- **Embedding**: LSB substitution
- **Fragile**: Any modification breaks watermark
- **Caching**: `@st.cache_data` for performance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Navigate through the sidebar:
1. **Embed**: Upload original image → Download watermarked PNG
2. **Attack**: Upload watermarked image → Apply attacks → Download attacked image
3. **Verify**: Upload suspect image → View tamper analysis

## Security Notes

- Change the default secret key for production use
- Use strong, unique secret keys
- Watermark is invisible but fragile to any modification
- PNG format recommended to avoid compression artifacts

## Dependencies

- streamlit: Web app framework
- streamlit-drawable-canvas: Interactive drawing canvas
- Pillow: Image processing
- numpy: Numerical operations
