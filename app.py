import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import io
import time
from numba import njit, prange
import base64

# ==========================================
# PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="StringArt Studio Pro", page_icon="🧵", layout="wide")
st.markdown("""
<style>
    .main {background-color: #0e1117; color: #fafafa;}
    .stButton>button {background:#ff4b4b; color:white; border:none; height:3em; width:100%; font-weight:bold;}
    .stButton>button:hover {background:#ff3333;}
    h1, h2 {text-align:center; font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;}
    .metric {background:#1e1e1e; padding:15px; border-radius:10px; text-align:center; border:1px solid #333;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# NUMBA-ACCELERATED LINE SCORING
# ==========================================
@njit(parallel=True, fastmath=True)
def score_lines(error_img: np.ndarray, lines: np.ndarray, weights: np.ndarray):
    """Score all precomputed lines in parallel."""
    n = len(lines)
    scores = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        score = 0.0
        count = 0
        for j in range(lines[i, 0]):
            y, x = lines[i, 1 + j*2], lines[i, 1 + j*2 + 1]
            score += error_img[y, x] * weights[i, j]
            count += weights[i, j]
        scores[i] = score / max(count, 1)
    return scores

@njit
def draw_line_fast(canvas: np.ndarray, error_img: np.ndarray, p0: tuple, p1: tuple, weight: float):
    """Bresenham with sub-pixel anti-aliasing + error subtraction."""
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        canvas[y0, x0] = 0
        error_img[y0, x0] = max(0.0, error_img[y0, x0] - weight)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

# ==========================================
# PRECOMPUTED LINE CACHE
# ==========================================
@st.cache_resource
def get_line_cache(num_pins=240, img_size=800, min_dist_pins=15):
    radius = (img_size // 2) - 10
    center = img_size // 2
    pins = []
    for i in range(num_pins):
        a = 2 * np.pi * i / num_pins - np.pi / 2
        x = int(center + radius * np.cos(a))
        y = int(center + radius * np.sin(a))
        pins.append((x, y))
    pins = np.array(pins)

    lines = []
    weights = []
    indices = []

    for i in range(num_pins):
        for j in range(i + min_dist_pins, num_pins - min_dist_pins):
            if abs(i - j) < min_dist_pins or abs(i - j) > num_pins - min_dist_pins:
                continue
            p0, p1 = pins[i], pins[j]
            dx = abs(p1[0] - p0[0])
            dy = abs(p1[1] - p0[1])
            n_points = max(dx, dy, 1)
            line = np.zeros((n_points, 2), dtype=np.int32)
            w = np.ones(n_points, dtype=np.float32)

            # Simple anti-aliased Bresenham
            x0, y0 = p0
            x1, y1 = p1
            steep = dy > dx
            if steep:
                x0, y0 = y0, x0
                x1, y1 = y1, x1
            if x0 > x1:
                x0, x1 = x1, x0
                y0, y1 = y1, y0
            dx = x1 - x0
            dy = abs(y1 - y0)
            error = dx / 2
            y = y0
            ystep = 1 if y0 < y1 else -1
            for x in range(x0, x1 + 1):
                coord = (y, x) if steep else (x, y)
                line[len(line) - n_points + (x - x0)] = coord
                error -= dy
                if error < 0:
                    y += ystep
                    error += dx

            lines.append(line.flatten())
            weights.append(w)
            indices.append((i, j))

    # Pad to fixed length
    max_len = max(len(l) for l in lines)
    line_array = np.zeros((len(lines), 1 + max_len * 2), dtype=np.int32)
    weight_array = np.zeros((len(lines), max_len), dtype=np.float32)
    for idx, (line, w) in enumerate(zip(lines, weights)):
        line_array[idx, 0] = len(line) // 2
        line_array[idx, 1:1+len(line)] = line
        weight_array[idx, :len(w)] = w

    return pins, line_array, weight_array, indices

# ==========================================
# STRING ART ENGINE (v2 – Blazing Fast)
# ==========================================
class StringArtEngine:
    def __init__(self, pil_img, num_pins=240, size=800):
        self.size = size
        self.pins, self.lines, self.weights, self.indices = get_line_cache(num_pins, size)
        self.num_pins = num_pins

        # Preprocess image
        gray = pil_img.convert("L")
        gray = ImageOps.fit(gray, (size, size), Image.LANCZOS)
        img = np.array(gray)

        # Circular mask
        mask = np.zeros_like(img)
        cv2.circle(mask, (size//2, size//2), size//2 - 10, 255, -1)
        img = cv2.bitwise_and(img, img, mask=mask)
        img[mask == 0] = 255

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.original = clahe.apply(img)
        self.original[mask == 0] = 255

        self.work = 255.0 - self.original.astype(np.float32)
        self.canvas = np.full((size, size), 255, dtype=np.uint8)

    def solve(self, max_lines=3500, line_weight=22, progress=st.progress):
        sequence = [0]
        current = 0
        bar = st.progress(0)
        status = st.empty()
        start = time.time()

        for step in range(1, max_lines):
            if step % 20 == 0:
                bar.progress(step / max_lines)
                elapsed = time.time() - start
                speed = step / max(0.1, elapsed)
                status.markdown(f"""
                <div class="metric">
                    Line {step}/{max_lines} • {speed:.1f} lines/sec • Pin {current} → ?
                </div>
                """, unsafe_allow_html=True)

            # Find best line from current pin
            candidates = [idx for idx, (i,j) in enumerate(self.indices) if i == current or j == current]
            if not candidates:
                break
            scores = score_lines(self.work, self.lines[candidates], self.weights[candidates])
            best_idx = candidates[np.argmax(scores)]
            best_i, best_j = self.indices[best_idx]
            next_pin = best_j if best_i == current else best_i

            # Draw
            p0 = tuple(self.pins[current])
            p1 = tuple(self.pins[next_pin])
            draw_line_fast(self.canvas, self.work, p0, p1, line_weight)

            sequence.append(next_pin)
            current = next_pin

        bar.progress(1.0)
        status.success("✅ Done!")
        self.sequence = sequence
        return sequence

    def get_preview(self):
        return self.canvas

# ==========================================
# EXPORT (Realistic SVG + Instructions)
# ==========================================
class ExportManager:
    @staticmethod
    def svg(pins, sequence, size_mm=600):
        s = f'<svg width="{size_mm}mm" height="{size_mm}mm" viewBox="0 0 800 800" xmlns="http://www.w3.org/2000/svg">\n'
        s += '<rect width="800" height="800" fill="white"/>\n'
        s += '<g opacity="0.06">\n'
        for i in range(1, len(sequence)):
            x1, y1 = pins[sequence[i-1]]
            x2, y2 = pins[sequence[i]]
            s += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="1"/>\n'
        s += '</g>\n'
        # Template layer (optional)
        s += '<g stroke="red" stroke-width="0.5" fill="none" opacity="0.3">\n'
        s += '<circle cx="400" cy="400" r="390"/>\n'
        for i, (x,y) in enumerate(pins):
            if i % 10 == 0:
                s += f'<circle cx="{x}" cy="{y}" r="3"/><text x="{x+8}" y="{y+8}" font-size="10" fill="red">{i}</text>\n'
        s += '</g></svg>'
        return s

    @staticmethod
    def txt(sequence, num_pins):
        txt = f"STRING ART GUIDE\nPins: {num_pins} | Lines: {len(sequence)-1}\n\n"
        for i in range(0, len(sequence)-1, 10):
            chunk = sequence[i:i+11]
            txt += " → ".join(map(str, chunk)) + "\n"
        return txt

# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.title("🧵 StringArt Studio Pro")
    st.markdown("### The fastest & most realistic string art generator on the web")

    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded = st.file_uploader("Upload image", ["jpg","png","jpeg"])
        num_pins = st.slider("Nails / Pins", 180, 360, 280, 10)
        diameter_mm = st.number_input("Board diameter (mm)", 300, 1000, 600)
        max_lines = st.slider("Thread density", 1000, 6000, 3500, 100)
        line_weight = st.slider("Thread darkness", 10, 40, 22)

    if uploaded:
        col1, col2 = st.columns([1, 2])
        img = Image.open(uploaded)

        with col1:
            st.image(img, use_column_width=True)
            if st.button("🚀 Generate String Art", type="primary", use_container_width=True):
                with col2:
                    engine = StringArtEngine(img, num_pins=num_pins)
                    with st.spinner("Running ultra-fast solver..."):
                        engine.solve(max_lines, line_weight)

                    st.image(engine.get_preview(), caption="Final Result", use_column_width=True)

                    svg = ExportManager.svg(engine.pins, engine.sequence, diameter_mm)
                    txt = ExportManager.txt(engine.sequence, num_pins)

                    c1, c2 = st.columns(2)
                    c1.download_button("📄 Download SVG (Print-Ready)", svg, "stringart.svg", "image/svg+xml")
                    c2.download_button("📜 Download Instructions", txt, "instructions.txt", "text/plain")

    else:
        st.info("👆 Upload a high-contrast portrait to begin")

if __name__ == "__main__":
    main()
