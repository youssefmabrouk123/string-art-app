import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import time

st.set_page_config(page_title="String Art Pro", layout="wide", page_icon="🧵")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #eee; }
    .stButton>button { background-color: #ff4b4b; color: white; font-weight: bold; border-radius: 8px; }
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

class StringArtGenerator:
    def __init__(self, num_pins=250, size=600):
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 5
        self.center = (size // 2, size // 2)
        
        # Precompute Pins
        self.pins = []
        for i in range(num_pins):
            angle = 2 * np.pi * i / num_pins - np.pi/2 # Start at top
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))
            
        # CACHE: Precompute line coordinates to speed up the loop 100x
        # Using a dictionary of tuples
        self.line_cache = {}
        
    def precompute_lines(self):
        """Calculates pixel coordinates for all possible pin connections"""
        # Only need to compute for the upper triangle of the matrix (undirected graph)
        for i in range(self.num_pins):
            for j in range(i + 1, self.num_pins):
                x0, y0 = self.pins[i]
                x1, y1 = self.pins[j]
                
                # Bresenham-like generation using OpenCV (Faster than Python loop)
                # We draw a line on a blank mask to get coordinates
                # This is a hack to get C++ speed for line coords
                mask = np.zeros((self.size, self.size), dtype=np.uint8)
                cv2.line(mask, (x0, y0), (x1, y1), 255, 1)
                
                # Extract coordinates
                # Note: numpy returns (row, col) -> (y, x)
                ys, xs = np.nonzero(mask)
                self.line_cache[(i, j)] = (ys, xs)
                self.line_cache[(j, i)] = (ys, xs) # Bi-directional

    def preprocess_image(self, pil_image, contrast=1.2, brightness=10, edge_enhance=False):
        """
        Converts user photo into a 'String Density Map'.
        Dark areas = High values (255). Light areas = Low values (0).
        """
        # 1. Grayscale & Resize
        img = pil_image.convert('L')
        img = ImageOps.fit(img, (self.size, self.size), Image.Resampling.LANCZOS)
        
        # 2. Edge Enhancement (Optional - mimic Michael Crum style)
        # This helps defined eyes and facial features
        if edge_enhance:
            edges = img.filter(ImageFilter.FIND_EDGES)
            img = Image.blend(img, edges, 0.2) # Blend 20% edges in
            
        # 3. Convert to Numpy
        arr = np.array(img, dtype=np.float32)
        
        # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Essential for fixing lighting issues in photos
        arr_uint8 = arr.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        arr_uint8 = clahe.apply(arr_uint8)
        arr = arr_uint8.astype(np.float32)
        
        # 5. Brightness/Contrast manual tweak
        # Formula: New = (Old - 127.5) * Contrast + 127.5 + Brightness
        arr = (arr - 127.5) * contrast + 127.5 + brightness
        arr = np.clip(arr, 0, 255)
        
        # 6. MASKING (Circle)
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2)
        mask = dist <= self.radius
        
        # 7. INVERSION logic
        # Algorithm seeks 'Max Value'. We want black string on white paper.
        # So Dark pixels in photo should be HIGH (255) in calculation map.
        # Standard photo: White=255, Black=0.
        # Inverted: White=0, Black=255.
        
        final_map = np.zeros_like(arr)
        final_map[mask] = 255 - arr[mask] # Invert inside circle
        
        self.error_matrix = final_map
        return 255 - final_map.astype(np.uint8) # Return visual preview (White background)

    def solve(self, max_lines=3000, line_weight=25, min_dist=20):
        """
        The Greedy Algorithm.
        line_weight: How much 'darkness' to remove per line. 
                     LOWER = Better shading (Gradient). 
                     HIGHER = Faster, rougher look.
        """
        # Ensure lines are computed
        if not self.line_cache:
            self.precompute_lines()
            
        sequence = [0]
        current_pin = 0
        
        # Visualization stuff
        bar = st.progress(0)
        status = st.empty()
        
        # Work on a copy so we can re-run without reloading
        work_matrix = self.error_matrix.copy()
        
        total_pins = self.num_pins
        
        # MAIN LOOP
        for step in range(max_lines):
            best_pin = -1
            best_score = -1.0
            
            # OPTIMIZATION 1: Don't check neighbor pins (boring lines)
            # We iterate through all valid targets
            candidates = []
            for i in range(total_pins):
                dist = abs(i - current_pin)
                if dist > min_dist and dist < (total_pins - min_dist):
                    candidates.append(i)
            
            # OPTIMIZATION 2: Check every 2nd pin to speed up loop (optional)
            # candidates = candidates[::2] 
            
            # Evaluate candidates
            for target in candidates:
                ys, xs = self.line_cache[(current_pin, target)]
                
                # Calculate score: Sum of pixel intensities along the line
                if len(xs) == 0: continue
                
                # Using SUM allows focusing on very dark spots. 
                # Using MEAN (Average) focuses on overall darkness.
                # Standard algo uses Mean.
                line_vals = work_matrix[ys, xs]
                score = np.mean(line_vals)
                
                if score > best_score:
                    best_score = score
                    best_pin = target
            
            # If we found a line
            if best_pin != -1:
                sequence.append(best_pin)
                
                # SUBTRACT Error
                ys, xs = self.line_cache[(current_pin, best_pin)]
                
                # We subtract the line_weight from the working matrix.
                # This means "we have covered this darkness with a thread".
                # Using maximum(0, ...) ensures we don't go into negative numbers.
                work_matrix[ys, xs] = np.maximum(0, work_matrix[ys, xs] - line_weight)
                
                current_pin = best_pin
            else:
                break
            
            if step % 50 == 0:
                bar.progress(min(step/max_lines, 1.0))
                status.text(f"Weaving line {step}/{max_lines}")
                
        bar.empty()
        status.empty()
        return sequence

    def render_realistic(self, sequence, opacity_percent=15):
        """
        Renders a PNG that simulates real thread physics using Alpha Blending.
        opacity_percent: 0-100. Lower = more realistic shading.
        """
        scale = 2
        w = self.size * scale
        
        # 1. White Canvas
        img = Image.new("RGB", (w, w), "white")
        draw = ImageDraw.Draw(img, "RGBA")
        
        # 2. Thread Color (Black with Alpha)
        alpha = int((255 * opacity_percent) / 100)
        thread_color = (0, 0, 0, alpha)
        
        # 3. Draw Frame
        cx, cy = self.center[0]*scale, self.center[1]*scale
        r = self.radius * scale
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=(200,200,200), width=3)
        
        # 4. Draw Pins (optional, makes it look techy)
        scaled_pins = [(x*scale, y*scale) for x,y in self.pins]
        # for px, py in scaled_pins:
        #     draw.ellipse((px-2, py-2, px+2, py+2), fill=(100,100,100))
            
        # 5. Draw Lines
        # Drawing thousands of alpha lines in PIL can be slow, but looks best.
        for i in range(len(sequence)-1):
            p0 = scaled_pins[sequence[i]]
            p1 = scaled_pins[sequence[i+1]]
            draw.line([p0, p1], fill=thread_color, width=2)
            
        # 6. Resize for Anti-Aliasing
        img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)
        return img

    def create_svg(self, sequence):
        svg = [f'<svg height="{self.size}mm" width="{self.size}mm" viewBox="0 0 {self.size} {self.size}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<circle cx="{self.size/2}" cy="{self.size/2}" r="{self.radius}" stroke="#ccc" fill="none"/>')
        
        # Optimized Polyline (One single object instead of thousands of lines)
        points = []
        for p in sequence:
            x, y = self.pins[p]
            points.append(f"{x},{y}")
            
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="black" stroke-width="0.25" opacity="0.7" />')
        svg.append('</svg>')
        return "\n".join(svg)

# ================= UI =================

st.title("🧵 String Art Studio")
st.caption("High-Fidelity Algorithm (Michael Crum Style)")

col_left, col_right = st.columns([1, 1.5], gap="medium")

with col_left:
    uploaded_file = st.file_uploader("Upload Portrait", type=['jpg', 'png'])
    
    st.write("---")
    st.subheader("Tuning")
    
    # INPUT TUNING
    contrast = st.slider("Contrast Boost", 0.5, 2.5, 1.3, help="Increase this to make the face pop out.")
    brightness = st.slider("Brightness", -50, 50, 0)
    
    # ALGO TUNING
    st.write("---")
    num_pins = st.slider("Pin Count", 200, 360, 250)
    max_lines = st.slider("Max Lines", 2000, 5000, 3500, help="More lines = smoother shading.")
    
    # THE SECRET SAUCE
    line_weight = st.slider("Thread Weight (Opacity)", 10, 100, 30, 
                            help="CRITICAL: Keep this LOW (20-40) for detailed shading. If High, image turns black fast.")
    
    run_btn = st.button("GENERATE ART", type="primary", use_container_width=True)

with col_right:
    if uploaded_file:
        # Initialize
        image = Image.open(uploaded_file)
        gen = StringArtGenerator(num_pins=num_pins, size=600)
        
        # Preprocess Preview
        preview_arr = gen.preprocess_image(image, contrast, brightness)
        
        # Tabs for viewing
        tab1, tab2 = st.tabs(["👁️ Algorithm View", "🎨 Final Result"])
        
        with tab1:
            st.image(preview_arr, caption="What the Computer Sees (Must be high contrast!)", use_column_width=True)
            st.info("Tip: Adjust Contrast until the background is White and eyes are Black.")
            
        with tab2:
            if run_btn:
                with st.spinner("Calculating Geometry..."):
                    # 1. Compute
                    t0 = time.time()
                    sequence = gen.solve(max_lines=max_lines, line_weight=line_weight)
                    t1 = time.time()
                    
                    # 2. Render
                    final_img = gen.render_realistic(sequence, opacity_percent=15) # Render slightly transparent for screen
                    
                    st.success(f"Done in {t1-t0:.2f}s | {len(sequence)} lines")
                    st.image(final_img, use_column_width=True)
                    
                    # 3. Downloads
                    svg_data = gen.create_svg(sequence)
                    txt_data = f"Pins: {num_pins}\nSequence: " + "->".join(map(str, sequence))
                    
                    c1, c2 = st.columns(2)
                    c1.download_button("Download SVG", svg_data, "art.svg", "image/svg+xml")
                    c2.download_button("Download Steps", txt_data, "guide.txt", "text/plain")
    else:
        st.info("Please upload an image to start.")
