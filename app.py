# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import cv2
from io import BytesIO
import time

# --- DEFAULT SAMPLE PATH (from your upload) ---
SAMPLE_PATH = "/mnt/data/4ec9b995-e446-4b8a-b07a-64a644b73a3e.png"

st.set_page_config(page_title="HQ String Art", layout="wide", page_icon="🧵")
st.title("HQ String Art — Use sample or upload")
st.sidebar.header("Settings")

# UI controls
pins = st.sidebar.slider("Pins (ring nails)", 200, 600, 420, step=10)
size = st.sidebar.selectbox("Canvas size (square)", [650, 800, 900, 1024], index=2)
max_lines = st.sidebar.slider("Max lines (iterations)", 5000, 40000, 25000, step=1000)
thickness = st.sidebar.slider("Render thread thickness", 1, 6, 2)
preview_every = st.sidebar.slider("Preview every N lines", 200, 800, 800)

uploaded = st.file_uploader("Upload portrait (jpg/png/webp). If none, sample will load.", type=["jpg","jpeg","png","webp"])

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

# -------------------------
# Main
# -------------------------
# load image (use uploaded if provided otherwise sample path)
if uploaded:
    img = Image.open(uploaded)
else:
    try:
        img = Image.open(SAMPLE_PATH)
    except Exception:
        st.error("Sample not found and no upload provided.")
        st.stop()

st.image(img, caption="Input", use_column_width=False, width=320)

# create algorithm instance
pins_coords, center, radius = create_pins(pins, size)
algo = HQStringArt(pins, size, center, radius)
algo.make_pins()
algo.preprocess(img)
with st.expander("Preview preprocessed target (inverted):", expanded=False):
    st.image(np.clip(algo.target, 0, 255).astype(np.uint8), use_column_width=False, width=320)

# precompute lines (may take a few seconds)
with st.spinner("Precomputing lines..."):
    t0 = time.time()
    algo.precompute_lines()
    t1 = time.time()
    st.text(f"Line cache ready — computed in {t1-t0:.1f}s — cached {len(algo.line_cache)//2} unique lines")

# progress UI elements
progress_bar = st.progress(0)
status = st.empty()
preview_slot = st.empty()

def progress_cb(it, max_iters, darkness, score):
    progress_bar.progress(min(it / max_iters, 1.0))
    status.text(f"Iter {it}/{max_iters} — Darkness {darkness:.0f} — Best score {score:.1f}")

def preview_cb(img_arr, it):
    preview_slot.image(img_arr, caption=f"Preview — {it} lines", use_column_width=False, width=420)

# run solver
run_button = st.button("Run solver")
if run_button:
    start = time.time()
    seq = algo.solve(max_iters=max_lines, progress_callback=progress_cb, preview_callback=preview_cb, preview_every=preview_every)
    end = time.time()
    st.success(f"Solved {len(seq)-1} lines in {end-start:.1f}s")
    # final preview from algorithm's canvas
    st.image(algo.final, caption="Algorithm reconstruction (grayscale)", use_column_width=False, width=420)

    # render high-quality image
    with st.spinner("Rendering final vector-like image..."):
        rendered = algo.render(thickness=thickness)
        st.image(rendered, caption="Final Render", use_column_width=False, width=640)

        # download
        buf = BytesIO()
        rendered.save(buf, format="PNG")
        st.download_button("Download PNG", buf.getvalue(), "string_art_result.png", "image/png")

# hint if nothing run
if not run_button:
    st.info("Adjust settings and press 'Run solver'. Use fewer lines for quick tests, increase for final high-quality results.")
