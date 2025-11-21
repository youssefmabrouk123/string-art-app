import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import time

st.set_page_config(page_title="🧵 Pro String Art", layout="wide", page_icon="🧵")

st.markdown("""
<style>
    .stApp { background-color: #0e0e0e; color: #eee; }
    .stButton>button { background-color: #00cc66; color: white; font-weight: bold; border-radius: 12px; padding: 12px; }
    h1, h2, h3 { color: #00ff88 !important; }
</style>
""", unsafe_allow_html=True)

class ProStringArt:
    def __init__(self, num_pins=250, size=600):
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 10
        self.center = (size // 2, size // 2)

    def setup(self):
        angles = np.linspace(0, 2*np.pi, self.num_pins, endpoint=False)
        self.pins = []
        for a in angles:
            x = int(self.center[0] + self.radius * np.cos(a))
            y = int(self.center[1] + self.radius * np.sin(a))
            self.pins.append((x, y))

        # Precompute all valid lines (faster + cleaner)
        self.lines = {}
        for i in range(self.num_pins):
            for j in range(i+1, self.num_pins):
                pixels = self._bresenham(self.pins[i], self.pins[j])
                if len(pixels) > 20:  # ignore tiny lines
                    self.lines[(i,j)] = self.lines[(j,i)] = pixels

        # Circular weight mask: center stronger, edges fade
        y, x = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        self.weight_mask = np.clip(1.0 - (dist / self.radius)**2, 0.3, 1.0)  # gentle falloff

    def _bresenham(self, p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        pixels = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            if 0 <= x0 < self.size and 0 <= y0 < self.size:
                pixels.append((y0, x0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return pixels

    def preprocess(self, pil_img, contrast=3.5, brightness=10, invert=True):
        img = pil_cv = np.array(pil_img.convert('L'))

        # Extreme CLAHE for dramatic contrast
        clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8,8))
        img = clahe.apply(img)

        # Brightness & final invert
        img = np.clip(img.astype(float) + brightness, 0, 255).astype(np.uint8)
        if invert:
            img = 255 - img

        # Circular mask
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        img = cv2.bitwise_and(img, img, mask=mask)

        # Final source: high = needs thread
        self.source = img.astype(np.float64)
        self.display = img.copy()
        return img

    def solve(self, max_lines=5500, base_darkness=190, threshold=12.0):
        self.setup()
        work = self.source.copy()
        output = np.full((self.size, self.size), 255.0)

        sequence = [0]
        current = 0
        taboo = set()  # recently used pins (anti-clump)

        progress = st.progress(0)
        status = st.empty()
        preview = st.empty()

        for step in range(1, max_lines):
            best_score = 0
            best_pin = -1
            darkness = base_darkness * (1.0 - step / max_lines * 0.5)  # decay

            for cand in range(self.num_pins):
                if cand in taboo: continue
                dist = min(abs(cand - current), self.num_pins - abs(cand - current))
                if dist < 18 + (step > 3000) * 15: continue  # adaptive skip

                key = tuple(sorted([current, cand]))
                if key not in self.lines: continue
                pixels = self.lines[key]

                # Weighted score: favor center
                score = sum(work[y,x] * self.weight_mask[y,x] for y,x in pixels) / len(pixels)

                if score > best_score:
                    best_score = score
                    best_pin = cand

            if best_pin == -1 or best_score < threshold:
                break

            # Add line
            sequence.append(best_pin)
            pixels = self.lines[tuple(sorted([current, best_pin]))]

            for y,x in pixels:
                work[y,x] = max(0, work[y,x] - darkness)
                output[y,x] -= darkness / 5.2  # visual darkness

            # Update taboo
            taboo = {current, best_pin}
            if len(taboo) > 12: taboo.pop()

            current = best_pin

            if step % 80 == 0:
                progress.progress(step / max_lines)
                status.text(f"Line {step} • Score {best_score:.1f} • Darkness {darkness:.0f}")

            if step % 500 == 0:
                prev = np.clip(output, 0, 255).astype(np.uint8)
                preview.image(prev, width=450, caption=f"Preview — {step} lines")

        progress.progress(1.0)
        st.success(f"Finished • {len(sequence)-1} lines")

        self.sequence = sequence
        self.output = np.clip(output, 0, 255).astype(np.uint8)
        return sequence

    def render_final(self):
        scale = 3
        sz = self.size * scale
        canvas = Image.new('RGBA', (sz, sz), (255,255,255,255))
        draw = ImageDraw.Draw(canvas)

        pins = [(x*scale, y*scale) for x,y in self.pins]
        cx, cy = self.center[0]*scale, self.center[1]*scale
        r = self.radius * scale

        # Frame
        draw.ellipse((cx-r-10, cy-r-10, cx+r+10, cy+r+10), outline=(80,80,80), width=12)

        # Threads — semi-transparent black
        for i in range(len(self.sequence)-1):
            p0 = pins[self.sequence[i]]
            p1 = pins[self.sequence[i+1]]
            draw.line([p0, p1], fill=(0,0,0,23), width=4)

        # Downsample with Lanczos for silky look
        final = canvas.resize((self.size, self.size), Image.Resampling.LANCZOS)
        rgb = Image.new('RGB', final.size, (255,255,255))
        rgb.paste(final, mask=final.split()[3])
        return rgb

# ================== UI ==================

st.title("🧵 Professional String Art Generator")
st.caption("Now matches real Petros Vrellis quality • Updated Nov 2025")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload Portrait", ['jpg','jpeg','png','webp'])
    if uploaded:
        orig = Image.open(uploaded)
        st.image(orig, caption="Original", width=300)

    st.markdown("### Best Settings (just click Generate)")
    pins = st.select_slider("Pins", options=[200,220,240,250,260,280,300], value=250)
    lines = st.select_slider("Lines", options=[3000,4000,4500,5000,5500,6000], value=5500)
    generate = st.button("🚀 GENERATE PROFESSIONAL STRING ART", type="primary", use_container_width=True)

with col2:
    if uploaded and generate:
        with st.spinner("Creating masterpiece..."):
            art = ProStringArt(num_pins=pins, size=600)
            art.preprocess(orig, contrast=3.5, brightness=15, invert=True)
            art.solve(max_lines=lines, base_darkness=190, threshold=14)

            final = art.render_final()
            st.image(final, caption=f"✓ {len(art.sequence)-1} threads • Professional quality", use_column_width=True)

            # Downloads
            buf = BytesIO()
            final.save(buf, 'PNG')
            svg = art.get_svg() if hasattr(art, 'get_svg') else None

            c1, c2 = st.columns(2)
            c1.download_button("🖼️ Download PNG", buf.getvalue(), "string_art_pro.png", "image/png")
            if svg:
                c2.download_button("📐 Download SVG", svg, "string_art.svg", "image/svg+xml")

            st.balloons()
    elif uploaded:
        st.info("👆 Click GENERATE above for professional result")
    else:
        st.info("Upload a clear portrait • Works best with good lighting and contrast")
        st.image("https://i.imgur.com/5eRjWQh.jpg", caption="This level of quality is now standard")  # example
