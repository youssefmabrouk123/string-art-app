import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
from io import BytesIO

st.set_page_config(page_title="Perfect String Art", layout="wide", page_icon="🧵")
st.title("🧵 Perfect String Art — Professional Generator")


# ---------------------------------------------------------
# CORE STRING-ART ENGINE
# ---------------------------------------------------------
class StringArt:
    def __init__(self, pins=260, size=700):
        self.pins = pins
        self.size = size
        self.radius = size // 2 - 10
        self.center = (size // 2, size // 2)

    # Pin placement on circle
    def make_pins(self):
        angles = np.linspace(0, 2*np.pi, self.pins, endpoint=False)
        X = (self.center[0] + self.radius * np.cos(angles)).astype(int)
        Y = (self.center[1] + self.radius * np.sin(angles)).astype(int)
        return list(zip(X, Y))

    # Image preprocessing: strong but smooth contrast + hair/face enhancement
    def preprocess(self, img):
        img = img.convert("L")
        img = ImageOps.fit(img, (self.size, self.size), Image.LANCZOS)
        arr = np.array(img)

        # CLAHE makes eyes/hair extremely sharp
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        arr = clahe.apply(arr)

        # Slight gamma boost (amazing difference)
        arr = ((arr / 255.0) ** 0.8 * 255).astype(np.uint8)

        # Invert into “darkness map”
        arr = 255 - arr

        # Circular mask
        mask = np.zeros_like(arr)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        arr = cv2.bitwise_and(arr, mask)

        self.target = arr.astype(np.float32)
        self.work = arr.astype(np.float32)

    # Fast Bresenham line
    def line_points(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((y0, x0))
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    # Main solver
    def solve(self, max_lines=7200):
        pins = self.make_pins()
        sequence = []
        current = 0

        # Canvas starts white
        canvas = np.ones((self.size, self.size), dtype=np.float32) * 255

        for n in range(max_lines):
            best_pin = None
            best_score = -1
            best_points = None

            # Dynamic darkness curve — avoids black face
            darkness = 120 - 50 * (n / max_lines)

            # Search for best line
            for offset in range(15, self.pins // 2):
                for direction in (1, -1):
                    cand = (current + direction * offset) % self.pins
                    x0, y0 = pins[current]
                    x1, y1 = pins[cand]

                    pts = self.line_points(x0, y0, x1, y1)
                    vals = [self.work[y, x] for (y, x) in pts]

                    score = np.mean(vals)
                    if score > best_score:
                        best_score = score
                        best_pin = cand
                        best_points = pts

            # Stop if no meaningful improvement
            if best_score < 4:
                break

            # Apply thread (VERY IMPORTANT: soft subtract)
            for (y, x) in best_points:
                subtract = min(darkness, self.work[y, x] * 0.85)
                self.work[y, x] -= subtract
                canvas[y, x] -= subtract / 6

            sequence.append(best_pin)
            current = best_pin

            if (n + 1) % 300 == 0:
                st.write(f"Lines: {n+1}")

        self.sequence = sequence
        self.canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # Final rendering using smooth alpha threads
    def render(self):
        scale = 4
        W = self.size * scale
        result = Image.new("RGBA", (W, W), (255, 255, 255, 255))
        draw = ImageDraw.Draw(result)

        pins_big = [(x*scale, y*scale) for (x, y) in self.make_pins()]

        for i in range(len(self.sequence) - 1):
            a = pins_big[self.sequence[i]]
            b = pins_big[self.sequence[i + 1]]
            draw.line([a, b], fill=(0, 0, 0, 28), width=5)

        out = result.resize((self.size, self.size), Image.LANCZOS)
        white = Image.new("RGB", out.size, (255, 255, 255))
        white.paste(out, mask=out.split()[3])
        return white


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload portrait", ["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Original", use_column_width=True)

    if st.button("Generate Perfect String Art", type="primary"):
        with st.spinner("Processing…"):
            engine = StringArt(pins=260, size=700)
            engine.preprocess(img)
            engine.solve()
            final = engine.render()

            st.image(final, caption="String Art Result", use_column_width=True)

            buf = BytesIO()
            final.save(buf, "PNG", quality=95)
            st.download_button("Download PNG", buf.getvalue(),
                               "string_art.png", "image/png")

else:
    st.info("Upload a portrait to start.")
