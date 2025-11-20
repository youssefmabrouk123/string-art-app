import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import time

st.set_page_config(page_title="String Art V8", layout="wide", page_icon="🧵")

st.markdown("""
<style>
    .stApp { background-color: #111; color: #eee; }
    .stButton>button { background-color: #00cc66; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class StringArt:
    def __init__(self, num_pins=200, size=500):
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 2
        self.center = (size // 2, size // 2)
        
    def setup(self):
        """Generate pins and precompute all lines"""
        # Generate pins around circle
        self.pins = []
        for i in range(self.num_pins):
            angle = 2 * np.pi * i / self.num_pins
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))
        
        # Precompute ALL line pixels using Bresenham
        self.lines = {}
        for i in range(self.num_pins):
            for j in range(i + 1, self.num_pins):
                pixels = self._bresenham(self.pins[i], self.pins[j])
                self.lines[(i, j)] = pixels
                self.lines[(j, i)] = pixels
    
    def _bresenham(self, p0, p1):
        """Bresenham line algorithm"""
        x0, y0 = p0
        x1, y1 = p1
        pixels = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if 0 <= x0 < self.size and 0 <= y0 < self.size:
                pixels.append((y0, x0))  # row, col for numpy
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pixels
    
    def preprocess(self, pil_img, contrast=2.0, brightness=0):
        """Prepare image: grayscale, CLAHE, circular mask"""
        # Convert and resize
        img = pil_img.convert('L')
        img = ImageOps.fit(img, (self.size, self.size), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        
        # CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        
        # Brightness
        arr = np.clip(arr.astype(float) + brightness, 0, 255).astype(np.uint8)
        
        # Circular mask - outside = WHITE
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        arr = np.where(mask == 255, arr, 255).astype(np.uint8)
        
        self.display_img = arr.copy()
        
        # INVERT: algorithm needs HIGH value = DARK area (needs strings)
        # 0 = white (ignore), 255 = black (cover)
        self.source = (255 - arr).astype(np.float64)
        
        return arr
    
    def solve(self, max_lines=4000, attenuation=0.5, min_skip=20, 
              threshold=5.0, progress_ui=None):
        """
        Greedy algorithm:
        1. Find line with highest average pixel value in source
        2. Draw it on output
        3. ATTENUATE source pixels along line (multiply by factor)
        4. Repeat
        """
        self.setup()
        
        # Working image - we reduce values as we add strings
        work = self.source.copy()
        
        # Output canvas (white = 255)
        output = np.ones((self.size, self.size), dtype=np.float64) * 255.0
        
        sequence = [0]  # Start at pin 0
        current = 0
        
        progress = st.progress(0)
        status = st.empty()
        preview = st.empty()
        
        line_count = 0
        
        for step in range(max_lines):
            best_pin = -1
            best_score = 0.0
            
            # Check all valid destination pins
            for cand in range(self.num_pins):
                # Skip nearby pins
                dist = min(abs(cand - current), self.num_pins - abs(cand - current))
                if dist < min_skip:
                    continue
                
                # Get line pixels
                key = (current, cand)
                if key not in self.lines:
                    continue
                pixels = self.lines[key]
                if len(pixels) < 10:
                    continue
                
                # SCORE = average pixel value along line
                # High value = dark area = good place for string
                total = sum(work[p[0], p[1]] for p in pixels)
                score = total / len(pixels)
                
                if score > best_score:
                    best_score = score
                    best_pin = cand
            
            # Stop if no improvement or below threshold
            if best_pin == -1 or best_score < threshold:
                status.text(f"Converged at {step} lines (score: {best_score:.1f})")
                break
            
            # Add line
            sequence.append(best_pin)
            pixels = self.lines[(current, best_pin)]
            
            # CRITICAL: Attenuate source image along line
            # This marks pixels as "covered" so algorithm moves on
            # Using multiplication (0.5 = halving) works better than subtraction
            for py, px in pixels:
                work[py, px] *= attenuation
                # Also darken output
                output[py, px] = max(0, output[py, px] - 30)
            
            current = best_pin
            line_count += 1
            
            # UI updates
            if step % 100 == 0:
                progress.progress(min(step / max_lines, 0.99))
                status.text(f"Line {step}/{max_lines} | Score: {best_score:.1f}")
            
            # Preview every 500 lines
            if step % 500 == 0 and step > 0:
                prev_img = np.clip(output, 0, 255).astype(np.uint8)
                preview.image(prev_img, caption=f"Preview @ {step} lines", width=400)
        
        progress.progress(1.0)
        time.sleep(0.3)
        progress.empty()
        status.empty()
        preview.empty()
        
        self.sequence = sequence
        self.output = np.clip(output, 0, 255).astype(np.uint8)
        self.line_count = line_count
        
        return sequence
    
    def render_hd(self, opacity=25, width=1):
        """High quality render with transparency blending"""
        scale = 2
        sz = self.size * scale
        
        canvas = Image.new('RGBA', (sz, sz), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        
        pins = [(p[0]*scale, p[1]*scale) for p in self.pins]
        
        # Frame
        cx, cy = self.center[0]*scale, self.center[1]*scale
        r = self.radius * scale
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=(100, 100, 100), width=3)
        
        # Pins
        for px, py in pins:
            draw.ellipse((px-3, py-3, px+3, py+3), fill=(60, 60, 60))
        
        # Strings
        color = (0, 0, 0, opacity)
        for i in range(len(self.sequence) - 1):
            p0 = pins[self.sequence[i]]
            p1 = pins[self.sequence[i+1]]
            draw.line([p0, p1], fill=color, width=width*scale)
        
        # Downscale
        canvas = canvas.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        rgb = Image.new('RGB', canvas.size, (255, 255, 255))
        rgb.paste(canvas, mask=canvas.split()[3])
        return rgb
    
    def get_svg(self):
        lines = [
            f'<svg width="{self.size}" height="{self.size}" viewBox="0 0 {self.size} {self.size}" xmlns="http://www.w3.org/2000/svg">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<circle cx="{self.center[0]}" cy="{self.center[1]}" r="{self.radius}" stroke="#aaa" fill="none"/>',
        ]
        for i, (px, py) in enumerate(self.pins):
            lines.append(f'<circle cx="{px}" cy="{py}" r="2" fill="#333"/>')
        
        path = f'M {self.pins[self.sequence[0]][0]} {self.pins[self.sequence[0]][1]}'
        for idx in self.sequence[1:]:
            path += f' L {self.pins[idx][0]} {self.pins[idx][1]}'
        lines.append(f'<path d="{path}" fill="none" stroke="black" stroke-width="0.3" opacity="0.5"/>')
        lines.append('</svg>')
        return '\n'.join(lines)
    
    def get_instructions(self):
        txt = [
            "STRING ART GUIDE",
            "=" * 40,
            f"Pins: {self.num_pins}",
            f"Lines: {len(self.sequence)-1}",
            "",
            "SEQUENCE:",
        ]
        for i in range(0, len(self.sequence), 20):
            chunk = self.sequence[i:i+20]
            txt.append(" → ".join(str(p) for p in chunk))
        return '\n'.join(txt)


# ========== UI ==========

st.title("🧵 String Art Generator V8")
st.caption("Fixed algorithm with proper attenuation (multiply by 0.5)")

c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("📤 Upload")
    up = st.file_uploader("Image", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if up:
        img = Image.open(up)
        st.image(img, caption="Original", use_column_width=True)
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    pins = st.slider("Pins", 100, 300, 200)
    lines = st.slider("Max Lines", 2000, 6000, 4000, step=500)
    
    with st.expander("Advanced"):
        size = st.selectbox("Size", [400, 500, 600], index=1)
        contrast = st.slider("CLAHE", 1.0, 4.0, 2.0)
        brightness = st.slider("Brightness", -50, 50, 0)
        attenuation = st.slider("Attenuation", 0.3, 0.7, 0.5,
                               help="0.5 = halve pixel value after each line")
        min_skip = st.slider("Min Pin Gap", 10, 40, 20)
        threshold = st.slider("Stop Threshold", 1.0, 20.0, 5.0)
        opacity = st.slider("Render Opacity", 15, 40, 25)
    
    go = st.button("🚀 GENERATE", type="primary", use_container_width=True)

with c2:
    if up and go:
        st.subheader("Processing...")
        
        art = StringArt(num_pins=pins, size=size)
        
        # Preprocess
        proc = art.preprocess(img, contrast=contrast, brightness=brightness)
        
        col_a, col_b = st.columns(2)
        col_a.image(proc, caption="Preprocessed", use_column_width=True)
        col_b.image(255 - proc, caption="Target (white=strings needed)", use_column_width=True)
        
        # Solve
        st.subheader("🧮 Computing...")
        t0 = time.time()
        seq = art.solve(max_lines=lines, attenuation=attenuation, 
                       min_skip=min_skip, threshold=threshold)
        elapsed = time.time() - t0
        
        # Render
        st.subheader("🎨 Result")
        final = art.render_hd(opacity=opacity)
        st.image(final, caption=f"{len(seq)-1} strings | {elapsed:.1f}s", use_column_width=True)
        
        # Compare
        st.subheader("📊 Comparison")
        x1, x2 = st.columns(2)
        x1.image(img, caption="Original", use_column_width=True)
        x2.image(final, caption="String Art", use_column_width=True)
        
        # Downloads
        st.subheader("📥 Downloads")
        d1, d2, d3 = st.columns(3)
        
        svg = art.get_svg()
        d1.download_button("📐 SVG", svg, "art.svg")
        
        txt = art.get_instructions()
        d2.download_button("📋 Guide", txt, "guide.txt")
        
        from io import BytesIO
        buf = BytesIO()
        final.save(buf, format='PNG')
        d3.download_button("🖼️ PNG", buf.getvalue(), "art.png")
        
        st.success(f"✅ {len(seq)-1} lines | {pins} pins | {elapsed:.1f}s")
        
    elif up:
        st.info("👈 Click GENERATE")
    else:
        st.info("👈 Upload an image")
        st.markdown("""
        ### Conseils pour de bons résultats:
        - **Visage centré** qui remplit 70%+ de l'image
        - **Bon contraste** entre le sujet et le fond
        - **Fond simple** (blanc ou uni)
        - **Éclairage clair** avec des ombres définies
        
        ### Paramètres recommandés:
        | Paramètre | Valeur |
        |-----------|--------|
        | Pins | 200-250 |
        | Lines | 4000-5000 |
        | Attenuation | 0.4-0.6 |
        """)
