# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import cv2
from io import BytesIO
import time

# --- DEFAULT SAMPLE PATH (from your upload) ---
SAMPLE_PATH = "/mnt/data/4ec9b995-e446-4b8a-b07a-64a644b73a3e.png"

# -------------------------
# Utility functions
# -------------------------
def to_cv_gray(img_pil, target_size):
    img = img_pil.convert("L")
    img = ImageOps.fit(img, (target_size, target_size), Image.LANCZOS)
    return np.array(img).astype(np.float32)

def create_pins(count, size):
    radius = size // 2 - 12
    center = (size // 2, size // 2)
    ang = np.linspace(0, 2*np.pi, count, endpoint=False)
    return [(int(center[0] + radius * np.cos(a)), int(center[1] + radius * np.sin(a))) for a in ang], center, radius

def bresenham(x0, y0, x1, y1, size):
    pts = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= x0 < size and 0 <= y0 < size:
            pts.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pts

# -------------------------
# High-quality pipeline
# -------------------------
class HQStringArt:
    def __init__(self, pins_count, size, center, radius):
        self.pins_count = pins_count
        self.size = size
        self.center = center
        self.radius = radius
        self.pins = None
        self.line_cache = {}

    def make_pins(self):
        ang = np.linspace(0, 2*np.pi, self.pins_count, endpoint=False)
        self.pins = [(int(self.center[0] + self.radius * np.cos(a)),
                      int(self.center[1] + self.radius * np.sin(a))) for a in ang]
        return self.pins

    def preprocess(self, pil_img):
        arr = to_cv_gray(pil_img, self.size).astype(np.uint8)

        # stronger CLAHE and gamma to bring out midtones
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        arr = clahe.apply(arr)
        # gamma correction
        gamma = 0.65
        arr = np.clip(((arr / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        # invert so dark regions => higher target
        arr = 255 - arr

        # circular mask
        mask = np.zeros_like(arr, dtype=np.uint8)
        cv2.circle(mask, (self.center[0], self.center[1]), self.radius + 3, 255, -1)
        arr = arr * (mask // 255)

        # float canvas and working map
        self.target = arr.astype(np.float32)
        self.work = self.target.copy()
        # start canvas white (light) high values => lighter
        self.canvas = np.full_like(self.work, 255.0, dtype=np.float32)

    def precompute_lines(self):
        # Precompute line points for each pair but only for reasonable separations to save RAM
        self.line_cache = {}
        pins = self.pins
        n = self.pins_count
        min_offset = 12
        max_offset = n//2
        for i in range(n):
            for d in range(min_offset, max_offset):
                j = (i + d) % n
                key = (i, j)
                if key in self.line_cache:
                    continue
                x0, y0 = pins[i]
                x1, y1 = pins[j]
                pts = bresenham(x0, y0, x1, y1, self.size)
                if len(pts) >= 20:
                    self.line_cache[key] = pts
                    self.line_cache[(j, i)] = pts

    def solve(self, max_iters=20000, progress_callback=None, preview_callback=None, preview_every=800):
        seq = [0]
        cur = 0
        n = self.pins_count

        # adaptive darkness schedule: start stronger, slowly weaken
        for it in range(1, max_iters + 1):
            best_score = -1.0
            best_pin = None
            best_pts = None

            # tune search window: try offsets 12..n//2
            for d in range(12, n//2):
                cand = (cur + d) % n
                key = (cur, cand)
                pts = self.line_cache.get(key)
                if not pts:
                    continue
                # vectorized mean
                vals = [self.work[y, x] for (y, x) in pts]
                avg = (sum(vals) / len(vals))
                # center weighting: points closer to center count more
                # compute a small center weight boost (faster on darker center features)
                if avg > best_score:
                    best_score = avg
                    best_pin = cand
                    best_pts = pts
                # early accept
                if avg > 50:
                    break

            if best_pin is None:
                break

            # darkness schedule
            frac = min(it / (max_iters * 0.6), 1.0)
            darkness = 160 - 100 * (frac ** 1.3)  # from ~160 -> ~60

            # apply subtract proportional to local darkness but never overshoot
            for (y, x) in best_pts:
                curval = self.work[y, x]
                subtract = min(darkness, curval * 0.92)
                if subtract <= 0:
                    continue
                self.work[y, x] = max(curval - subtract, 0.0)
                # visual canvas receives partial ink (controls final render darkness)
                self.canvas[y, x] = max(self.canvas[y, x] - subtract / 4.2, 0.0)

            seq.append(best_pin)
            cur = best_pin

            if progress_callback and (it % 100 == 0 or it == 1):
                progress_callback(it, max_iters, darkness, best_score)

            if preview_callback and (it % preview_every == 0):
                preview = np.clip(self.canvas, 0, 255).astype(np.uint8)
                preview_callback(preview, it)

        self.sequence = seq
        self.final = np.clip(self.canvas, 0, 255).astype(np.uint8)
        return seq

    def render(self, thickness=2, scale=3):
        # high-quality RGBA render with accumulation of semi-opaque lines
        big = Image.new("RGBA", (self.size*scale, self.size*scale), (255,255,255,255))
        draw = ImageDraw.Draw(big, 'RGBA')
        scaled = [(x*scale, y*scale) for (x,y) in self.pins]
        alpha = max(10, 40 - thickness*6)  # thinner threads darker alpha
        for i in range(len(self.sequence)-1):
            a = scaled[self.sequence[i]]
            b = scaled[self.sequence[i+1]]
            draw.line([a, b], fill=(0,0,0,alpha), width=thickness*scale)
        final = big.resize((self.size, self.size), Image.LANCZOS)
        bg = Image.new("RGB", final.size, (255,255,255))
        bg.paste(final, mask=final.split()[3])
        return bg

# Configure page with luxury styling
st.set_page_config(
    page_title="Luxury String Art Studio", 
    layout="wide", 
    page_icon="🎨",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for luxury styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 300;
        letter-spacing: 2px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(45deg, #2c3e50, #4a6491);
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(45deg, #34495e, #5676a8);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    .progress-bar {
        height: 8px;
        border-radius: 4px;
    }
    .preview-box {
        border-radius: 10px;
        padding: 1rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #bdc3c7, transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown('<h1 class="main-header">LUXURY STRING ART STUDIO</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform portraits into elegant string art masterpieces</p>', unsafe_allow_html=True)

# Sidebar with elegant organization
st.sidebar.markdown("## 🎛️ Control Panel")

# Image upload section
st.sidebar.markdown("### Image Source")
uploaded = st.sidebar.file_uploader(
    "Upload portrait (jpg/png/webp)", 
    type=["jpg","jpeg","png","webp"],
    help="For best results, use high-contrast portraits with clear facial features"
)

# Algorithm parameters in an expander
with st.sidebar.expander("⚙️ Algorithm Parameters", expanded=True):
    pins = st.slider("Number of Pins", 200, 600, 420, step=10, 
                    help="More pins create finer details but increase processing time")
    size = st.selectbox("Canvas Size", [650, 800, 900, 1024], index=2,
                       help="Larger canvas for higher resolution results")
    max_lines = st.slider("Maximum Lines", 5000, 40000, 25000, step=1000,
                         help="More lines create darker, more detailed images")
    
with st.sidebar.expander("🎨 Rendering Options", expanded=True):
    thickness = st.slider("Thread Thickness", 1, 6, 2,
                         help="Thicker threads create bolder visual impact")
    preview_every = st.slider("Preview Frequency", 200, 800, 800, step=100,
                             help="How often to update the preview during generation")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Input Image")
    
    # Load image (use uploaded if provided otherwise sample path)
    if uploaded:
        img = Image.open(uploaded)
        st.success("✅ Custom image loaded successfully")
    else:
        try:
            img = Image.open(SAMPLE_PATH)
            st.info("ℹ️ Using sample image. Upload your own for custom results.")
        except Exception:
            st.error("Sample not found and no upload provided.")
            st.stop()
    
    # Display input image with elegant container
    with st.container():
        st.image(img, caption="Original Portrait", use_column_width=True)
    
    # Image metrics
    st.markdown("#### Image Details")
    col1a, col1b, col1c = st.columns(3)
    with col1a:
        st.metric("Dimensions", f"{img.size[0]} × {img.size[1]}")
    with col1b:
        st.metric("Mode", img.mode)
    with col1c:
        st.metric("Format", uploaded.type if uploaded else "PNG")

with col2:
    st.markdown("### Processing Preview")
    
    # Create algorithm instance
    pins_coords, center, radius = create_pins(pins, size)
    algo = HQStringArt(pins, size, center, radius)
    algo.make_pins()
    algo.preprocess(img)
    
    # Preprocessed image preview
    with st.expander("🔍 Preprocessed Target (Inverted)", expanded=False):
        st.image(np.clip(algo.target, 0, 255).astype(np.uint8), 
                use_column_width=True, 
                caption="Preprocessed image ready for string art generation")
    
    # Algorithm status
    st.markdown("#### Generation Status")
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Performance metrics placeholder
    metrics_placeholder = st.empty()

# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Control section
st.markdown("### Generation Controls")
control_col1, control_col2, control_col3 = st.columns([1, 2, 1])

with control_col2:
    run_button = st.button("🎨 Generate String Art", use_container_width=True)

# Only run precomputation if we have an image
if 'img' in locals():
    # Precompute lines (may take a few seconds)
    with st.spinner("🔄 Precomputing optimal pin connections..."):
        t0 = time.time()
        algo.precompute_lines()
        t1 = time.time()
        
        # Display precomputation metrics
        with metrics_placeholder.container():
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Precomputation Time", f"{t1-t0:.1f}s")
            with mcol2:
                st.metric("Unique Lines", f"{len(algo.line_cache)//2:,}")
            with mcol3:
                st.metric("Pin Count", pins)

# Preview slot for intermediate results
preview_slot = st.empty()

# Progress callback functions
def progress_cb(it, max_iters, darkness, score):
    progress_bar.progress(min(it / max_iters, 1.0))
    status_placeholder.text(f"🔄 Processing: {it:,}/{max_iters:,} lines | Darkness: {darkness:.0f} | Score: {score:.1f}")

def preview_cb(img_arr, it):
    with preview_slot.container():
        st.markdown(f"#### Progress Preview ({it:,} lines)")
        st.image(img_arr, use_column_width=True)

# Run solver when button is clicked
if run_button:
    start = time.time()
    
    # Create columns for side-by-side comparison
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.markdown("### Algorithm Reconstruction")
    
    with result_col2:
        st.markdown("### Final Render")
    
    # Run the algorithm
    seq = algo.solve(
        max_iters=max_lines, 
        progress_callback=progress_cb, 
        preview_callback=preview_cb, 
        preview_every=preview_every
    )
    
    end = time.time()
    
    # Completion message
    st.success(f"✅ Generation complete! {len(seq)-1:,} lines created in {end-start:.1f} seconds")
    
    # Display final results
    with result_col1:
        st.image(algo.final, 
                use_column_width=True, 
                caption=f"Algorithm reconstruction ({len(seq)-1:,} lines)")
    
    # Render high-quality image
    with result_col2:
        with st.spinner("🖌️ Applying final touches..."):
            rendered = algo.render(thickness=thickness)
            st.image(rendered, 
                    use_column_width=True, 
                    caption="Final string art masterpiece")
    
    # Download section
    st.markdown("---")
    st.markdown("### Download Your Artwork")
    
    col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
    
    with col_d2:
        # Prepare download
        buf = BytesIO()
        rendered.save(buf, format="PNG", optimize=True)
        
        # Elegant download button
        st.download_button(
            label="📥 Download High-Quality PNG",
            data=buf.getvalue(),
            file_name="luxury_string_art.png",
            mime="image/png",
            use_container_width=True
        )

# Initial instructions if no action taken
if not run_button:
    st.info("""
    **💡 Getting Started:** 
    1. Upload a portrait or use the sample image
    2. Adjust parameters in the sidebar for your desired effect
    3. Click 'Generate String Art' to create your masterpiece
    4. Download your artwork when complete
    """)
