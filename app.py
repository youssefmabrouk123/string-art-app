import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
from numba import jit, prange
import time

st.set_page_config(page_title="String Art Pro V7", layout="wide", page_icon="🧵")

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; color: #eee; }
    .stButton>button { background-color: #00cc66; color: white; font-weight: bold; border-radius: 8px; padding: 0.75rem 2rem; }
    .stProgress > div > div { background-color: #00cc66; }
</style>
""", unsafe_allow_html=True)

# ============ NUMBA OPTIMIZED FUNCTIONS ============

@jit(nopython=True, cache=True)
def bresenham_line(x0, y0, x1, y1, max_size):
    """Fast Bresenham line with Numba"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        if 0 <= x0 < max_size and 0 <= y0 < max_size:
            points.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

@jit(nopython=True, cache=True)
def compute_line_score(error_img, line_ys, line_xs, line_weight):
    """
    Compute squared error improvement for adding a line.
    Returns the total improvement (positive = good)
    """
    total_improvement = 0.0
    n = len(line_ys)
    
    for i in range(n):
        y, x = line_ys[i], line_xs[i]
        current_error = error_img[y, x]
        # New error after adding line (clamped to 0)
        new_error = max(0.0, current_error - line_weight)
        # Improvement = reduction in squared error
        improvement = current_error * current_error - new_error * new_error
        total_improvement += improvement
    
    return total_improvement

@jit(nopython=True, cache=True)
def apply_line(error_img, output_img, line_ys, line_xs, line_weight, output_weight):
    """Apply line to both error and output images"""
    n = len(line_ys)
    for i in range(n):
        y, x = line_ys[i], line_xs[i]
        error_img[y, x] = max(0.0, error_img[y, x] - line_weight)
        output_img[y, x] = max(0.0, output_img[y, x] - output_weight)

# ============ MAIN CLASS ============

class StringArtEngine:
    def __init__(self, num_pins=250, img_size=600):
        self.num_pins = num_pins
        self.img_size = img_size
        self.radius = (img_size // 2) - 2
        self.center = (img_size // 2, img_size // 2)
        
    def setup_pins(self):
        """Generate pins around circle"""
        self.pins = []
        for i in range(self.num_pins):
            angle = 2 * np.pi * i / self.num_pins
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))
        
        # Pre-compute all lines
        self._precompute_all_lines()
    
    def _precompute_all_lines(self):
        """Pre-compute all line pixels between pin pairs"""
        self.line_cache = {}
        
        for i in range(self.num_pins):
            for j in range(i + 1, self.num_pins):
                p0, p1 = self.pins[i], self.pins[j]
                pixels = bresenham_line(p0[0], p0[1], p1[0], p1[1], self.img_size)
                
                if len(pixels) > 0:
                    ys = np.array([p[0] for p in pixels], dtype=np.int32)
                    xs = np.array([p[1] for p in pixels], dtype=np.int32)
                    self.line_cache[(i, j)] = (ys, xs)
                    self.line_cache[(j, i)] = (ys, xs)
    
    def preprocess(self, pil_image, contrast=2.5, brightness=0, blur=1, gamma=1.0):
        """Prepare image for algorithm"""
        # Grayscale
        img = pil_image.convert('L')
        img = ImageOps.fit(img, (self.img_size, self.img_size), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        
        # Gaussian blur
        if blur > 0:
            arr = cv2.GaussianBlur(arr, (blur*2+1, blur*2+1), 0)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        
        # Gamma
        if gamma != 1.0:
            arr = np.power(arr / 255.0, gamma) * 255
            arr = arr.astype(np.uint8)
        
        # Brightness
        arr = np.clip(arr.astype(np.float32) + brightness, 0, 255).astype(np.uint8)
        
        # Circular mask - outside = WHITE (255)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        arr = np.where(mask == 255, arr, 255).astype(np.uint8)
        
        self.processed_display = arr.copy()
        
        # INVERT for algorithm: dark=high value
        # We want to REDUCE these high values by adding strings
        self.target_img = (255 - arr).astype(np.float64)
        
        return arr
    
    def compute(self, max_lines=4000, line_weight=25, min_skip=20, 
                allow_revisit=True, progress_placeholder=None):
        """
        Main algorithm using squared error minimization.
        Key insight: find line that maximizes reduction in squared error.
        """
        self.setup_pins()
        
        # Error image - what we need to cover (high value = needs string)
        error_img = self.target_img.copy()
        
        # Output image for visualization (starts white = 255)
        output_img = np.ones((self.img_size, self.img_size), dtype=np.float64) * 255.0
        
        sequence = [0]
        current_pin = 0
        
        # Track used lines (prevent immediate reuse)
        recent_lines = set()
        max_recent = 50 if not allow_revisit else 10
        
        # For convergence detection
        total_error = np.sum(error_img ** 2)
        stall_count = 0
        
        progress_bar = st.progress(0)
        status = st.empty()
        preview_placeholder = st.empty()
        
        output_weight = line_weight * 0.35  # Visual weight for rendering
        
        for step in range(max_lines):
            best_pin = -1
            best_score = 0.0
            
            # Search all valid pins
            for candidate in range(self.num_pins):
                # Skip nearby pins
                dist = min(abs(candidate - current_pin), 
                          self.num_pins - abs(candidate - current_pin))
                if dist < min_skip:
                    continue
                
                # Skip recently used
                line_key = (min(current_pin, candidate), max(current_pin, candidate))
                if line_key in recent_lines:
                    continue
                
                # Get line
                if (current_pin, candidate) not in self.line_cache:
                    continue
                ys, xs = self.line_cache[(current_pin, candidate)]
                
                if len(ys) < 10:
                    continue
                
                # Score = squared error improvement
                score = compute_line_score(error_img, ys, xs, line_weight)
                
                if score > best_score:
                    best_score = score
                    best_pin = candidate
            
            # Stop if no improvement
            if best_pin == -1 or best_score <= 0:
                status.text(f"Converged at {step} lines")
                break
            
            # Apply line
            sequence.append(best_pin)
            ys, xs = self.line_cache[(current_pin, best_pin)]
            apply_line(error_img, output_img, ys, xs, line_weight, output_weight)
            
            # Track recent lines
            line_key = (min(current_pin, best_pin), max(current_pin, best_pin))
            recent_lines.add(line_key)
            if len(recent_lines) > max_recent:
                recent_lines.pop()
            
            current_pin = best_pin
            
            # Progress updates
            if step % 200 == 0:
                progress_bar.progress(min(step / max_lines, 0.99))
                new_error = np.sum(error_img ** 2)
                reduction = (1 - new_error / total_error) * 100
                status.text(f"Line {step}/{max_lines} | Error reduced: {reduction:.1f}%")
                
                # Live preview every 500 lines
                if step % 500 == 0 and step > 0:
                    preview = np.clip(output_img, 0, 255).astype(np.uint8)
                    preview_placeholder.image(preview, caption=f"Preview @ {step} lines", 
                                             use_column_width=True)
        
        progress_bar.progress(1.0)
        preview_placeholder.empty()
        status.empty()
        progress_bar.empty()
        
        self.sequence = sequence
        self.output_img = np.clip(output_img, 0, 255).astype(np.uint8)
        return sequence
    
    def render_final(self, line_opacity=20, line_width=1):
        """High quality render with anti-aliasing"""
        scale = 2
        render_size = self.img_size * scale
        
        canvas = Image.new('RGBA', (render_size, render_size), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        
        scaled_pins = [(p[0] * scale, p[1] * scale) for p in self.pins]
        
        # Frame
        cx, cy = self.center[0] * scale, self.center[1] * scale
        rad = self.radius * scale
        draw.ellipse((cx-rad-2, cy-rad-2, cx+rad+2, cy+rad+2), 
                     outline=(120, 120, 120), width=3)
        
        # Nails
        for px, py in scaled_pins:
            draw.ellipse((px-3, py-3, px+3, py+3), fill=(50, 50, 50))
        
        # Strings
        thread_color = (0, 0, 0, line_opacity)
        for i in range(len(self.sequence) - 1):
            p0 = scaled_pins[self.sequence[i]]
            p1 = scaled_pins[self.sequence[i + 1]]
            draw.line([p0, p1], fill=thread_color, width=line_width * scale)
        
        # Downscale
        canvas = canvas.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        rgb = Image.new('RGB', canvas.size, (255, 255, 255))
        rgb.paste(canvas, mask=canvas.split()[3])
        return rgb
    
    def export_svg(self):
        lines = [
            f'<svg width="{self.img_size}" height="{self.img_size}" '
            f'viewBox="0 0 {self.img_size} {self.img_size}" xmlns="http://www.w3.org/2000/svg">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<circle cx="{self.center[0]}" cy="{self.center[1]}" r="{self.radius}" '
            f'stroke="#999" stroke-width="1" fill="none"/>',
        ]
        
        for i, (px, py) in enumerate(self.pins):
            lines.append(f'<circle cx="{px}" cy="{py}" r="2" fill="#333"/>')
        
        if len(self.sequence) > 1:
            path = f'M {self.pins[self.sequence[0]][0]} {self.pins[self.sequence[0]][1]}'
            for idx in self.sequence[1:]:
                path += f' L {self.pins[idx][0]} {self.pins[idx][1]}'
            lines.append(f'<path d="{path}" fill="none" stroke="black" '
                        f'stroke-width="0.3" stroke-opacity="0.5"/>')
        
        lines.append('</svg>')
        return '\n'.join(lines)
    
    def export_instructions(self):
        text = [
            "STRING ART INSTRUCTIONS",
            "=" * 50,
            f"Pins: {self.num_pins}",
            f"Connections: {len(self.sequence) - 1}",
            "",
            "SEQUENCE:",
            "-" * 50,
        ]
        for i in range(0, len(self.sequence), 20):
            chunk = self.sequence[i:i+20]
            text.append(" → ".join(f"{p:3d}" for p in chunk))
        return '\n'.join(text)


# ============ UI ============

st.title("🧵 String Art Generator V7")
st.caption("Optimized algorithm: squared error minimization + Numba acceleration")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📤 Upload Photo")
    uploaded = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if uploaded:
        orig = Image.open(uploaded)
        st.image(orig, caption="Original", use_column_width=True)
    
    st.markdown("---")
    st.subheader("⚙️ Parameters")
    
    num_pins = st.slider("Pins", 150, 350, 250)
    max_lines = st.slider("Max Lines", 2000, 8000, 4000, step=500,
                          help="More = denser image (4000-6000 recommended)")
    
    with st.expander("🔧 Advanced Settings"):
        img_size = st.selectbox("Resolution", [400, 500, 600, 700], index=2)
        contrast = st.slider("CLAHE Contrast", 1.0, 4.0, 2.5)
        gamma = st.slider("Gamma", 0.5, 2.0, 1.0, 
                         help="<1 = brighter, >1 = darker")
        brightness = st.slider("Brightness", -50, 50, 0)
        blur = st.slider("Blur", 0, 3, 1)
        line_weight = st.slider("Line Weight", 15, 50, 25,
                               help="How much each string 'uses'")
        min_skip = st.slider("Min Pin Gap", 10, 50, 20)
        render_opacity = st.slider("Render Opacity", 15, 35, 22)
    
    generate = st.button("🚀 GENERATE", use_container_width=True, type="primary")

with col_right:
    if uploaded and generate:
        st.subheader("Processing...")
        
        engine = StringArtEngine(num_pins=num_pins, img_size=img_size)
        
        # Preprocess
        processed = engine.preprocess(orig, contrast=contrast, brightness=brightness,
                                      blur=blur, gamma=gamma)
        
        col_a, col_b = st.columns(2)
        col_a.image(processed, caption="Preprocessed", use_column_width=True)
        col_b.image(255 - processed, caption="Target (dark = strings)", use_column_width=True)
        
        st.subheader("🧮 Computing...")
        start = time.time()
        
        sequence = engine.compute(
            max_lines=max_lines,
            line_weight=line_weight,
            min_skip=min_skip
        )
        
        elapsed = time.time() - start
        
        st.subheader("🎨 Result")
        final = engine.render_final(line_opacity=render_opacity)
        st.image(final, caption=f"{len(sequence)-1} strings in {elapsed:.1f}s", 
                use_column_width=True)
        
        # Side by side
        st.subheader("📊 Comparison")
        c1, c2 = st.columns(2)
        c1.image(orig, caption="Original", use_column_width=True)
        c2.image(final, caption="String Art", use_column_width=True)
        
        # Downloads
        st.subheader("📥 Downloads")
        d1, d2, d3 = st.columns(3)
        
        svg = engine.export_svg()
        d1.download_button("📐 SVG", svg, "string_art.svg")
        
        txt = engine.export_instructions()
        d2.download_button("📋 Guide", txt, "guide.txt")
        
        from io import BytesIO
        buf = BytesIO()
        final.save(buf, format='PNG')
        d3.download_button("🖼️ PNG", buf.getvalue(), "string_art.png")
        
        st.success(f"✅ {len(sequence)-1} connections | {num_pins} pins | {elapsed:.1f}s")
        
    elif uploaded:
        st.info("👈 Click GENERATE to start")
    else:
        st.info("👈 Upload a photo")
        st.markdown("""
        ### Pour de bons résultats:
        - **Visage centré** remplissant 70%+ de l'image
        - **Bon éclairage** avec ombres claires
        - **Fond simple** (uni ou flou)
        - **High contrast** entre sujet et fond
        
        ### Paramètres recommandés:
        - **Pins**: 200-300
        - **Lines**: 4000-6000
        - **Line Weight**: 20-30
        """)
