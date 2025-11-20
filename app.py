import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import time

st.set_page_config(page_title="String Art Studio V6", layout="wide", page_icon="🧵")

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; color: #eee; }
    .stButton>button { background-color: #00cc66; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class StringArtEngine:
    """
    Core algorithm based on research:
    - Greedy selection of darkest line
    - Subtract line contribution from source image
    - Stop when improvement drops below threshold
    """
    
    def __init__(self, num_pins=200, img_size=500):
        self.num_pins = num_pins
        self.img_size = img_size
        self.radius = (img_size // 2) - 2
        self.center = (img_size // 2, img_size // 2)
        self.pins = []
        self.line_cache = {}
        
    def setup_pins(self):
        """Generate pin positions around circle"""
        self.pins = []
        for i in range(self.num_pins):
            angle = 2 * np.pi * i / self.num_pins
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))
        
        # Pre-compute ALL lines between pins
        self._precompute_lines()
        
    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm - returns list of (row, col) tuples"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if 0 <= x0 < self.img_size and 0 <= y0 < self.img_size:
                points.append((y0, x0))  # (row, col) for numpy
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
    
    def _precompute_lines(self):
        """Pre-compute all line pixels between pin pairs"""
        self.line_cache = {}
        for i in range(self.num_pins):
            for j in range(i + 1, self.num_pins):
                p0, p1 = self.pins[i], self.pins[j]
                pixels = self._bresenham_line(p0[0], p0[1], p1[0], p1[1])
                # Store both directions
                self.line_cache[(i, j)] = pixels
                self.line_cache[(j, i)] = pixels
                
    def get_line(self, pin_a, pin_b):
        """Get cached line pixels"""
        return self.line_cache.get((pin_a, pin_b), [])
    
    def preprocess(self, pil_image, contrast=1.5, brightness=0):
        """
        Prepare image for algorithm:
        - Convert to grayscale
        - Apply CLAHE for local contrast
        - Apply circular mask
        - INVERT so dark pixels have HIGH values (what we want to cover)
        """
        # Grayscale
        img = pil_image.convert('L')
        
        # Resize to fit
        img = ImageOps.fit(img, (self.img_size, self.img_size), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        
        # CLAHE - Critical for face detail
        clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        
        # Brightness adjustment
        arr = np.clip(arr.astype(np.float32) + brightness, 0, 255).astype(np.uint8)
        
        # Circular mask - make outside WHITE (255) so algorithm ignores it
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        
        # Apply mask - outside circle becomes white
        result = np.where(mask == 255, arr, 255).astype(np.uint8)
        
        # INVERT: Algorithm needs dark areas = high values
        # Original: 0=black, 255=white
        # Inverted: 255=black (need strings), 0=white (ignore)
        self.source_img = (255 - result).astype(np.float64)
        
        return result  # Return non-inverted for display
    
    def compute(self, max_lines=3500, line_darkness=50, min_skip=20, 
                stop_threshold=5.0, progress_callback=None):
        """
        Main greedy algorithm:
        1. Find line with highest average darkness in source image
        2. Add that line to sequence
        3. SUBTRACT line contribution from source (attenuation)
        4. Repeat until improvement drops below threshold
        """
        self.setup_pins()
        
        # Working copy of source image
        # High value = dark area that needs strings
        work_img = self.source_img.copy()
        
        sequence = []
        current_pin = 0  # Start at pin 0
        sequence.append(current_pin)
        
        # For visualization - track what we've drawn
        output_img = np.ones((self.img_size, self.img_size), dtype=np.float64) * 255
        
        progress = st.progress(0)
        status = st.empty()
        
        used_lines = set()  # Prevent using same line twice in a row
        last_pins = []  # Track recent pins to avoid getting stuck
        
        for step in range(max_lines):
            best_pin = -1
            best_score = -1
            
            # Try all possible destination pins
            for candidate in range(self.num_pins):
                # Skip pins that are too close (creates short ugly lines)
                dist = min(abs(candidate - current_pin), 
                          self.num_pins - abs(candidate - current_pin))
                if dist < min_skip:
                    continue
                
                # Skip if we just used this exact line
                line_key = (min(current_pin, candidate), max(current_pin, candidate))
                if line_key in used_lines:
                    continue
                
                # Get line pixels
                pixels = self.get_line(current_pin, candidate)
                if len(pixels) < 10:
                    continue
                
                # SCORE = sum of pixel values along line in work image
                # Higher value = darker area = better line
                line_sum = sum(work_img[p[0], p[1]] for p in pixels)
                score = line_sum / len(pixels)  # Average darkness
                
                if score > best_score:
                    best_score = score
                    best_pin = candidate
            
            # Stop if no good line found or score too low
            if best_pin == -1 or best_score < stop_threshold:
                status.text(f"Converged at {step} lines (score: {best_score:.1f})")
                break
            
            # Add line to sequence
            sequence.append(best_pin)
            
            # Mark line as used (prevent immediate reuse)
            line_key = (min(current_pin, best_pin), max(current_pin, best_pin))
            used_lines.add(line_key)
            
            # Remove from used after some steps to allow reuse
            if len(used_lines) > 100:
                used_lines.pop()
            
            # CRITICAL: Attenuate (subtract) line from work image
            # This is the key insight - we "use up" the darkness
            pixels = self.get_line(current_pin, best_pin)
            for py, px in pixels:
                work_img[py, px] = max(0, work_img[py, px] - line_darkness)
                # Also darken output image
                output_img[py, px] = max(0, output_img[py, px] - line_darkness * 0.4)
            
            current_pin = best_pin
            
            # Progress update
            if step % 100 == 0:
                progress.progress(min(step / max_lines, 0.99))
                status.text(f"Line {step}/{max_lines} | Score: {best_score:.1f}")
        
        progress.progress(1.0)
        status.text(f"✅ Completed: {len(sequence)-1} connections")
        time.sleep(0.5)
        progress.empty()
        status.empty()
        
        self.sequence = sequence
        self.output_preview = np.clip(output_img, 0, 255).astype(np.uint8)
        return sequence
    
    def render_final(self, line_opacity=20):
        """
        Render high-quality output with proper alpha blending
        """
        # 2x resolution for anti-aliasing
        render_size = self.img_size * 2
        
        # RGBA for proper transparency blending
        canvas = Image.new('RGBA', (render_size, render_size), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        
        # Scale pins
        scaled_pins = [(p[0] * 2, p[1] * 2) for p in self.pins]
        
        # Draw frame
        cx, cy = self.center[0] * 2, self.center[1] * 2
        rad = self.radius * 2
        draw.ellipse((cx-rad, cy-rad, cx+rad, cy+rad), 
                     outline=(150, 150, 150), width=4)
        
        # Draw pins
        for i, (px, py) in enumerate(scaled_pins):
            draw.ellipse((px-4, py-4, px+4, py+4), fill=(80, 80, 80))
        
        # Draw strings with transparency
        thread_color = (0, 0, 0, line_opacity)  # Semi-transparent black
        
        for i in range(len(self.sequence) - 1):
            p0 = scaled_pins[self.sequence[i]]
            p1 = scaled_pins[self.sequence[i + 1]]
            draw.line([p0, p1], fill=thread_color, width=2)
        
        # Downscale with anti-aliasing
        canvas = canvas.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # Convert RGBA to RGB
        rgb = Image.new('RGB', canvas.size, (255, 255, 255))
        rgb.paste(canvas, mask=canvas.split()[3])
        
        return rgb
    
    def export_svg(self):
        """Generate SVG for cutting/printing"""
        lines = [
            f'<svg width="{self.img_size}mm" height="{self.img_size}mm" '
            f'viewBox="0 0 {self.img_size} {self.img_size}" xmlns="http://www.w3.org/2000/svg">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<circle cx="{self.center[0]}" cy="{self.center[1]}" r="{self.radius}" '
            f'stroke="#aaa" stroke-width="1" fill="none"/>',
        ]
        
        # Pin markers and numbers
        for i, (px, py) in enumerate(self.pins):
            lines.append(f'<circle cx="{px}" cy="{py}" r="2" fill="#333"/>')
        
        # Thread path
        if len(self.sequence) > 1:
            path = f'M {self.pins[self.sequence[0]][0]} {self.pins[self.sequence[0]][1]}'
            for idx in self.sequence[1:]:
                path += f' L {self.pins[idx][0]} {self.pins[idx][1]}'
            lines.append(f'<path d="{path}" fill="none" stroke="black" '
                        f'stroke-width="0.25" stroke-opacity="0.6"/>')
        
        lines.append('</svg>')
        return '\n'.join(lines)
    
    def export_instructions(self):
        """Human-readable winding guide"""
        text = [
            "=" * 50,
            "STRING ART WINDING INSTRUCTIONS",
            "=" * 50,
            f"Total Pins: {self.num_pins}",
            f"Total Connections: {len(self.sequence) - 1}",
            "",
            "PIN SEQUENCE (follow in order):",
            "-" * 50,
        ]
        
        # Format as groups
        for i in range(0, len(self.sequence), 15):
            chunk = self.sequence[i:i+15]
            text.append(f"[{i+1:4d}] " + " → ".join(f"{p:3d}" for p in chunk))
        
        return '\n'.join(text)


# =============== UI ===============

st.title("🧵 String Art Generator V6")
st.caption("Fixed algorithm with proper greedy search and attenuation")

# Two main columns
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📤 Upload")
    uploaded = st.file_uploader("Portrait photo", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if uploaded:
        orig_image = Image.open(uploaded)
        st.image(orig_image, caption="Original", use_column_width=True)
    
    st.markdown("---")
    st.subheader("⚙️ Parameters")
    
    # Key parameters
    num_pins = st.slider("Pins", 100, 300, 200, 
                         help="More pins = finer detail but longer compute")
    max_lines = st.slider("Max Lines", 1500, 5000, 3500,
                          help="More lines = darker, more detailed image")
    
    with st.expander("🔧 Advanced"):
        img_size = st.selectbox("Resolution", [400, 500, 600, 700], index=1)
        contrast = st.slider("CLAHE Contrast", 1.0, 4.0, 2.0)
        brightness = st.slider("Brightness", -50, 50, 0)
        line_darkness = st.slider("Line Darkness", 30, 80, 50,
                                  help="How much each string 'uses up' the image")
        min_skip = st.slider("Min Pin Gap", 10, 40, 20,
                            help="Minimum pins to skip (prevents short lines)")
        stop_threshold = st.slider("Stop Threshold", 1.0, 20.0, 5.0,
                                  help="Lower = more lines, higher = stops earlier")
        render_opacity = st.slider("Render Opacity", 15, 40, 22)
    
    generate = st.button("🚀 GENERATE", use_container_width=True, type="primary")

with col_right:
    if uploaded:
        if generate:
            st.subheader("🔄 Processing...")
            
            # Initialize engine
            engine = StringArtEngine(num_pins=num_pins, img_size=img_size)
            
            # Preprocess
            with st.spinner("Preprocessing image..."):
                processed = engine.preprocess(orig_image, contrast=contrast, brightness=brightness)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(processed, caption="Preprocessed (algorithm input)", use_column_width=True)
            with col_b:
                st.image(255 - processed, caption="Inverted (what algorithm sees as 'dark')", 
                        use_column_width=True)
            
            # Compute
            st.subheader("🧮 Computing String Path...")
            sequence = engine.compute(
                max_lines=max_lines,
                line_darkness=line_darkness,
                min_skip=min_skip,
                stop_threshold=stop_threshold
            )
            
            # Render
            st.subheader("🎨 Final Result")
            with st.spinner("Rendering high-quality output..."):
                final = engine.render_final(line_opacity=render_opacity)
            
            st.image(final, caption=f"String Art ({len(sequence)-1} strings)", use_column_width=True)
            
            # Compare side by side
            st.subheader("📊 Comparison")
            c1, c2 = st.columns(2)
            c1.image(orig_image, caption="Original", use_column_width=True)
            c2.image(final, caption="String Art", use_column_width=True)
            
            # Downloads
            st.subheader("📥 Downloads")
            d1, d2, d3 = st.columns(3)
            
            # SVG
            svg = engine.export_svg()
            d1.download_button("📐 SVG Vector", svg, "string_art.svg", "image/svg+xml")
            
            # Instructions
            txt = engine.export_instructions()
            d2.download_button("📋 Instructions", txt, "winding_guide.txt", "text/plain")
            
            # PNG
            from io import BytesIO
            buf = BytesIO()
            final.save(buf, format='PNG')
            d3.download_button("🖼️ PNG Image", buf.getvalue(), "string_art.png", "image/png")
            
            st.success(f"✅ Done! {len(sequence)-1} connections across {num_pins} pins")
            
        else:
            st.info("👈 Adjust parameters and click GENERATE")
    else:
        st.info("👈 Upload a portrait photo to begin")
        st.markdown("""
        ### Tips for best results:
        - **Face should fill 70%+** of the frame
        - **Good lighting** with clear shadows
        - **High contrast** between face and background
        - **Simple backgrounds** work best
        
        ### How it works:
        1. Image is converted to grayscale
        2. CLAHE enhances local contrast
        3. Greedy algorithm finds darkest lines
        4. Each line "uses up" darkness (attenuation)
        5. Process repeats until convergence
        """)
