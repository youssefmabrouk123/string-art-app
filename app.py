import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import cv2
from io import BytesIO

# =========================================================
#   UI CONFIG
# =========================================================
st.set_page_config(
    page_title="Perfect String Art Generator",
    layout="centered",
    page_icon="🧵"
)

st.title("🧵 Perfect String Art Generator")
st.caption("Upload a portrait → Get a flawless, high-detail string-art simulation")

# =========================================================
#   CORE CLASS
# =========================================================

class PerfectStringArt:
    def __init__(self, pins=260, size=650):
        self.pins_count = pins
        self.size = size
        self.radius = size // 2 - 15
        self.center = (size // 2, size // 2)

    def make_pins(self):
        angles = np.linspace(0, 2*np.pi, self.pins_count, endpoint=False)
        return [
            (
                int(self.center[0] + self.radius * np.cos(a)),
                int(self.center[1] + self.radius * np.sin(a))
            ) for a in angles
        ]

    def preprocess(self, img):
        img = img.convert("L")
        img = ImageOps.fit(img, (self.size, self.size), Image.LANCZOS)
        arr = np.array(img)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        arr = clahe.apply(arr)

        # Gentle brightness boost
        arr = np.clip(arr.astype(np.int16) + 20, 0, 255).astype(np.uint8)

        # Invert target
        arr = 255 - arr

        # Circular mask
        mask = np.zeros_like(arr)
        cv2.circle(mask, self.center, self.radius + 10, 255, -1)
        arr = cv2.bitwise_and(arr, arr, mask=mask)

        self.source = arr.astype(np.float32)
        self.original_dark = arr.copy()

    def fast_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if 0 <= x0 < self.size and 0 <= y0 < self.size:
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

    def solve(self, max_lines=7500):
        pins = self.make_pins()
        work = self.source.copy()
        canvas = np.full_like(work, 255.0)

        sequence = [0]
        current = 0

        progress = st.progress(0)
        status = st.empty()
        preview = st.empty()

        for line in range(1, max_lines):

            best_score = 0
            best_pin = None

            # Dynamic darkness reduction
            progress_factor = np.clip(line / 3000, 0, 1)
            darkness = 180 - 80 * progress_factor**1.5

            min_dist = 20 + int(15 * (line > 3500))

            # Search candidate pins
            for offset in range(15, self.pins_count // 2):
                for direction in (-1, 1):
                    cand = (current + direction * offset) % self.pins_count
                    if min(abs(cand - current), self.pins_count - abs(cand - current)) < min_dist:
                        continue

                    x0, y0 = pins[current]
                    x1, y1 = pins[cand]
                    pts = self.fast_line(x0, y0, x1, y1)
                    if len(pts) < 30:
                        continue

                    scores = [work[y, x] for y, x in pts]
                    if not scores:
                        continue

                    score = sum(scores) / len(scores)
                    if score > best_score:
                        best_score = score
                        best_pin = cand
                        best_pts = pts

                    if score > 40:
                        break
                if best_score > 40:
                    break

            if best_pin is None or best_score < 8:
                status.text(f"Finished at {line} lines")
                break

            # Apply reduction
            for y, x in best_pts:
                subtract = min(darkness, work[y, x] * 0.9)
                work[y, x] -= subtract
                canvas[y, x] -= subtract / 4.8

            sequence.append(best_pin)
            current = best_pin

            if line % 100 == 0:
                progress.progress(min(line / max_lines, 1.0))
                status.text(f"Line {line:,} — Darkness {darkness:.0f} — Score {best_score:.1f}")

            if line % 800 == 0:
                prev = np.clip(canvas, 0, 255).astype(np.uint8)
                preview.image(prev, caption=f"Preview — {line} lines", use_column_width=True)

        self.sequence = sequence
        self.final_canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        progress.progress(1.0)

    def render(self):
        scale = 4
        big = Image.new("RGBA", (self.size * scale, self.size * scale), (255, 255, 255, 255))
        draw = ImageDraw.Draw(big)

        scaled_pins = [(x * scale, y * scale) for x, y in self.make_pins()]

        for i in range(len(self.sequence) - 1):
            a = scaled_pins[self.sequence[i]]
            b = scaled_pins[self.sequence[i + 1]]
            draw.line([a, b], fill=(0, 0, 0, 24), width=6)

        final = big.resize((self.size, self.size), Image.LANCZOS)
        bg = Image.new("RGB", final.size, (255, 255, 255))
        bg.paste(final, mask=final.split()[3])

        return bg

# =========================================================
#   UI
# =========================================================

uploaded = st.file_uploader("Upload your portrait", ["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Original", use_column_width=True)

    if st.button("🧵 Generate String Art", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            art = PerfectStringArt()
            art.preprocess(img)
            art.solve()
            result = art.render()

            st.image(result, use_column_width=True)

            buf = BytesIO()
            result.save(buf, "PNG")
            st.download_button(
                "Download Result",
                buf.getvalue(),
                "string_art.png",
                "image/png"
            )
else:
    st.info("Upload a photo to begin.")
