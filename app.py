import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import time

st.set_page_config(page_title="String Art FINAL", layout="wide", page_icon="🧵")

st.markdown("""
<style>
    .stApp { background-color: #111; color: #eee; }
    .stButton>button { background-color: #00cc66; color: white; font-weight: bold; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

class StringArt:
    """
    Algorithm based on Petros Vrellis / kmmeerts / callummcdougall:
    - Image rescaled: WHITE=0, BLACK=255 (high value = needs string)
    - Find line with highest average darkness
    - Subtract 150-200 from pixels along line
    - Repeat until no improvement
    """
    
    def __init__(self, num_pins=200, size=500):
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 2
        self.center = (size // 2, size // 2)
        
    def setup(self):
        """Generate pins and precompute ALL lines"""
        # Pins around circle
        self.pins = []
        for i in range(self.num_pins):
            angle = 2 * np.pi * i / self.num_pins
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))
        
        # Precompute all line pixels
        self.lines = {}
        for i in range(self.num_pins):
            for j in range(i + 1, self.num_pins):
                pixels = self._line_pixels(self.pins[i], self.pins[j])
                self.lines[(i, j)] = pixels
                self.lines[(j, i)] = pixels
    
    def _line_pixels(self, p0, p1):
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
                pixels.append((y0, x0))
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
    
    def preprocess(self, pil_img, contrast=2.5, brightness=0, invert_input=False):
        """
        Prepare image for algorithm.
        OUTPUT: Array where BLACK=255, WHITE=0
        (high value = area that needs strings)
        """
        # Grayscale
        img = pil_img.convert('L')
        img = ImageOps.fit(img, (self.size, self.size), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        
        # Brightness
        arr = np.clip(arr.astype(float) + brightness, 0, 255).astype(np.uint8)
        
        # Optional invert (for photos with dark background)
        if invert_input:
            arr = 255 - arr
        
        # Circular mask - outside = WHITE (0 after inversion)
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        arr = np.where(mask == 255, arr, 255).astype(np.uint8)
        
        self.display_img = arr.copy()
        
        # INVERT: WHITE=0, BLACK=255
        # High value = dark area = needs strings
        self.source = (255 - arr).astype(np.float64)
        
        return arr
    
    def solve(self, max_lines=4000, darkness=150, min_skip=20, threshold=10.0):
        """
        Main algorithm:
        1. Find line with highest average pixel value (darkest area)
        2. Draw line on output
        3. SUBTRACT darkness (150-200) from source pixels
        4. Repeat until score < threshold
        """
        self.setup()
        
        # Working copy - high values = need strings
        work = self.source.copy()
        
        # Output canvas for real-time preview (white=255)
        output = np.ones((self.size, self.size), dtype=np.float64) * 255.0
        
        sequence = [0]
        current = 0
        
        progress = st.progress(0)
        status = st.empty()
        preview_col = st.empty()
        
        # For rendering output
        output_darkness = 30  # How dark each line appears in output
        
        for step in range(max_lines):
            best_pin = -1
            best_score = -1.0
            
            # Find best next pin
            for cand in range(self.num_pins):
                # Skip nearby pins (prevents short lines)
                dist = min(abs(cand - current), self.num_pins - abs(cand - current))
                if dist < min_skip:
                    continue
                
                # Get line pixels
                if (current, cand) not in self.lines:
                    continue
                pixels = self.lines[(current, cand)]
                if len(pixels) < 10:
                    continue
                
                # SCORE = average pixel value along line
                total = sum(work[p[0], p[1]] for p in pixels)
                score = total / len(pixels)
                
                if score > best_score:
                    best_score = score
                    best_pin = cand
            
            # Stop if no good line or below threshold
            if best_pin == -1 or best_score < threshold:
                status.text(f"✅ Converged at {step} lines (score={best_score:.1f})")
                break
            
            # Add line to sequence
            sequence.append(best_pin)
            pixels = self.lines[(current, best_pin)]
            
            # CRITICAL: Subtract darkness from source image
            # This marks pixels as "covered" - values between 150-200 work best
            for py, px in pixels:
                work[py, px] = max(0, work[py, px] - darkness)
                # Also update output visualization
                output[py, px] = max(0, output[py, px] - output_darkness)
            
            current = best_pin
            
            # Progress updates
            if step % 100 == 0:
                progress.progress(min(step / max_lines, 0.99))
                status.text(f"Line {step}/{max_lines} | Score: {best_score:.1f}")
            
            # Live preview
            if step % 300 == 0 and step > 0:
                prev = np.clip(output, 0, 255).astype(np.uint8)
                preview_col.image(prev, caption=f"Preview @ {step} lines", width=400)
        
        progress.progress(1.0)
        time.sleep(0.3)
        progress.empty()
        preview_col.empty()
        
        self.sequence = sequence
        self.output = np.clip(output, 0, 255).astype(np.uint8)
        
        return sequence
    
    def render_final(self, opacity=20, line_width=1):
        """High quality render with proper alpha blending"""
        scale = 2
        sz = self.size * scale
        
        # RGBA canvas
        canvas = Image.new('RGBA', (sz, sz), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        
        # Scale pins
        pins = [(p[0]*scale, p[1]*scale) for p in self.pins]
        
        # Frame circle
        cx, cy = self.center[0]*scale, self.center[1]*scale
        r = self.radius * scale
        draw.ellipse((cx-r-3, cy-r-3, cx+r+3, cy+r+3), outline=(100,100,100), width=4)
        
        # Draw pins
        for px, py in pins:
            draw.ellipse((px-4, py-4, px+4, py+4), fill=(50, 50, 50))
        
        # Draw strings with transparency
        # Each string is semi-transparent black - overlapping = darker
        thread_color = (0, 0, 0, opacity)
        
        for i in range(len(self.sequence) - 1):
            p0 = pins[self.sequence[i]]
            p1 = pins[self.sequence[i+1]]
            draw.line([p0, p1], fill=thread_color, width=line_width*scale)
        
        # Downscale with anti-aliasing
        canvas = canvas.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Convert RGBA to RGB
        rgb = Image.new('RGB', canvas.size, (255, 255, 255))
        rgb.paste(canvas, mask=canvas.split()[3])
        return rgb
    
    def get_svg(self):
        """Export SVG for laser cutting / plotting"""
        lines = [
            f'<svg width="{self.size}" height="{self.size}" '
            f'viewBox="0 0 {self.size} {self.size}" xmlns="http://www.w3.org/2000/svg">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<circle cx="{self.center[0]}" cy="{self.center[1]}" r="{self.radius}" '
            f'stroke="#888" stroke-width="2" fill="none"/>',
        ]
        
        # Pins
        for i, (px, py) in enumerate(self.pins):
            lines.append(f'<circle cx="{px}" cy="{py}" r="3" fill="#333"/>')
        
        # Thread path
        if len(self.sequence) > 1:
            path = f'M {self.pins[self.sequence[0]][0]} {self.pins[self.sequence[0]][1]}'
            for idx in self.sequence[1:]:
                path += f' L {self.pins[idx][0]} {self.pins[idx][1]}'
            lines.append(f'<path d="{path}" fill="none" stroke="black" '
                        f'stroke-width="0.4" stroke-opacity="0.6"/>')
        
        lines.append('</svg>')
        return '\n'.join(lines)
    
    def get_instructions(self):
        """Export winding guide"""
        txt = [
            "=" * 50,
            "STRING ART WINDING GUIDE",
            "=" * 50,
            f"Total Pins: {self.num_pins}",
            f"Total Connections: {len(self.sequence) - 1}",
            "",
            "Follow this sequence, wrapping thread around each pin:",
            "-" * 50,
        ]
        
        for i in range(0, len(self.sequence), 20):
            chunk = self.sequence[i:i+20]
            txt.append(" → ".join(str(p) for p in chunk))
        
        return '\n'.join(txt)


# ================== UI ==================

st.title("🧵 String Art Generator - FIXED")
st.caption("Algorithm: subtract 150-200 darkness per line (based on Petros Vrellis method)")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader("Choose photo", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if uploaded:
        original = Image.open(uploaded)
        st.image(original, caption="Original", use_column_width=True)
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    pins = st.slider("Number of Pins", 150, 300, 200)
    lines = st.slider("Max Lines", 2000, 6000, 4000, step=500)
    
    with st.expander("🔧 Advanced Settings", expanded=True):
        size = st.selectbox("Resolution", [400, 500, 600], index=1)
        contrast = st.slider("CLAHE Contrast", 1.0, 4.0, 2.5, 
                            help="Higher = more local contrast")
        brightness = st.slider("Brightness Adjust", -80, 80, 0)
        darkness = st.slider("Line Darkness", 100, 200, 150,
                            help="Amount subtracted per line. 150-200 recommended")
        min_skip = st.slider("Min Pin Distance", 15, 50, 25)
        threshold = st.slider("Stop Threshold", 5.0, 30.0, 15.0,
                             help="Lower = more lines, higher = stop earlier")
        render_opacity = st.slider("Render Thread Opacity", 15, 40, 22)
        invert = st.checkbox("Invert Input Image", False,
                            help="Check if your subject is LIGHT on DARK background")
    
    generate = st.button("🚀 GENERATE STRING ART", type="primary", use_container_width=True)

with col2:
    if uploaded and generate:
        st.subheader("🔄 Processing...")
        
        # Create engine
        art = StringArt(num_pins=pins, size=size)
        
        # Preprocess
        processed = art.preprocess(original, contrast=contrast, 
                                   brightness=brightness, invert_input=invert)
        
        # Show preprocessing
        col_a, col_b = st.columns(2)
        col_a.image(processed, caption="After CLAHE", use_column_width=True)
        col_b.image(art.source.astype(np.uint8), 
                   caption="Algorithm Input (WHITE=strings needed)", 
                   use_column_width=True)
        
        # Solve
        st.subheader("🧮 Computing String Path...")
        t0 = time.time()
        sequence = art.solve(max_lines=lines, darkness=darkness, 
                            min_skip=min_skip, threshold=threshold)
        elapsed = time.time() - t0
        
        # Render final
        st.subheader("🎨 Final Result")
        final = art.render_final(opacity=render_opacity)
        st.image(final, caption=f"{len(sequence)-1} strings | {elapsed:.1f}s", 
                use_column_width=True)
        
        # Comparison
        st.subheader("📊 Comparison")
        c1, c2 = st.columns(2)
        c1.image(original, caption="Original", use_column_width=True)
        c2.image(final, caption="String Art Result", use_column_width=True)
        
        # Downloads
        st.subheader("📥 Download Files")
        d1, d2, d3 = st.columns(3)
        
        svg = art.get_svg()
        d1.download_button("📐 SVG Vector", svg, "string_art.svg", "image/svg+xml")
        
        txt = art.get_instructions()
        d2.download_button("📋 Winding Guide", txt, "instructions.txt", "text/plain")
        
        from io import BytesIO
        buf = BytesIO()
        final.save(buf, format='PNG', quality=95)
        d3.download_button("🖼️ PNG Image", buf.getvalue(), "string_art.png", "image/png")
        
        st.success(f"✅ Complete! {len(sequence)-1} string connections with {pins} pins")
        
    elif uploaded:
        st.info("👈 Adjust settings and click GENERATE")
        
    else:
        st.info("👈 Upload an image to begin")
        
        st.markdown("""
        ### 🎯 Pour de bons résultats:
        
        **Image idéale:**
        - Visage centré, remplit 70%+ de l'image
        - Bon contraste entre sujet et fond
        - Fond simple (blanc/uni de préférence)
        - Éclairage clair avec ombres définies
        
        **Paramètres clés:**
        
        | Paramètre | Recommandé | Effet |
        |-----------|------------|-------|
        | **Pins** | 200-250 | Plus = plus de détails |
        | **Lines** | 4000-5000 | Plus = image plus dense |
        | **Darkness** | 150-180 | Contrôle la "consommation" |
        | **CLAHE** | 2.0-3.0 | Améliore le contraste local |
        
        ### 🔧 Comment ça marche:
        
        1. L'image est convertie: **NOIR=255** (besoin de fil), **BLANC=0** (ignorer)
        2. L'algorithme trouve la ligne avec la plus haute moyenne de pixels
        3. Il soustrait **150-200** de chaque pixel sur cette ligne
        4. Il répète jusqu'à ce qu'aucune amélioration ne soit possible
        """)
