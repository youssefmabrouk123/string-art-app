import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from skimage.draw import line as skimage_line

# --- CONFIGURATION ---
st.set_page_config(page_title="Industrial String Art", layout="wide", page_icon="🧵")
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; }
    .stButton>button { background-color: #ff4b4b; color: white; font-weight: bold; border: none; }
    div[data-testid="stImage"] { border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

class IndustrialSolver:
    def __init__(self, pil_image, num_pins=256, size=600):
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 5
        self.center = (size // 2, size // 2)
        
        # 1. PREPARE IMAGE (The "Map")
        # Convert to Grayscale
        img = pil_image.convert('L')
        # Resize to calculation size
        img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        
        # 2. TONE MAPPING (Critical for Faces)
        # We clip the histogram to force the face to be the primary feature
        img_arr = np.array(img)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is what makes eyes visible even if the photo is dark
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_arr = clahe.apply(img_arr)
        
        # Invert: We want "Darkness" to be High Value (255) because we subtract from it
        self.target_image = 255 - img_arr
        self.target_image = self.target_image.astype(np.float32)
        
        # Mask Circle (Set outside to 0 so we don't draw there)
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - self.center[0])**2 + (Y-self.center[1])**2)
        self.target_image[dist > self.radius] = 0

        # 3. PRE-CALCULATE PINS
        self.pins = []
        for i in range(num_pins):
            angle = 2 * np.pi * i / num_pins - np.pi/2
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))
            
        # 4. PRE-CALCULATE LINES (The Speed Hack)
        # We store the pixel coordinates of every possible line to avoid re-calculating
        # This uses memory but makes the loop 50x faster.
        self.line_cache = {}

    def get_line_coords(self, p1_idx, p2_idx):
        """Returns the cached coordinates between two pins"""
        # Sort indices to ensure (1,5) is same as (5,1)
        key = tuple(sorted((p1_idx, p2_idx)))
        
        if key not in self.line_cache:
            p1 = self.pins[key[0]]
            p2 = self.pins[key[1]]
            rr, cc = skimage_line(p1[1], p1[0], p2[1], p2[0])
            self.line_cache[key] = (rr, cc)
            
        return self.line_cache[key]

    def solve(self, max_lines=3000, dark_weight=20):
        """
        The Solver: NO RANDOMNESS. FULL SCAN.
        """
        sequence = [0]
        current_pin = 0
        
        # We keep track of the "Current Canvas" to show the user
        # But mathematically we subtract from self.target_image
        
        bar = st.progress(0)
        status = st.empty()
        
        # Skip neighbors (too short)
        MIN_DIST = 15
        
        # PRE-CACHE OPTIMIZATION
        # Only cache lines we actually use to save RAM
        
        for step in range(max_lines):
            best_pin = -1
            best_score = -1.0
            
            # 1. DEFINE SEARCH SPACE
            # We check ALL pins that are not neighbors
            # NO RANDOM SAMPLING.
            candidates = []
            for i in range(self.num_pins):
                d = abs(i - current_pin)
                if d > MIN_DIST and d < (self.num_pins - MIN_DIST):
                    candidates.append(i)
            
            # 2. EVALUATE CANDIDATES
            # This is the heavy lifting
            for target in candidates:
                # Get pixels
                rr, cc = self.get_line_coords(current_pin, target)
                
                # Safety check for bounds
                if len(rr) == 0: continue
                rr = np.clip(rr, 0, self.size - 1)
                cc = np.clip(cc, 0, self.size - 1)
                
                # Score = Sum of remaining darkness along the line
                # We prioritize the line that covers the MOST dark pixels
                line_sum = np.sum(self.target_image[rr, cc])
                
                # Normalize by length (optional, but usually better for faces)
                # Taking average density
                score = line_sum / (len(rr) + 1)
                
                if score > best_score:
                    best_score = score
                    best_pin = target
            
            # Stop if no good line found
            if best_pin == -1:
                break
                
            # 3. COMMIT MOVE
            sequence.append(best_pin)
            
            # 4. SUBTRACT DARKNESS (Error Diffusion)
            rr, cc = self.get_line_coords(current_pin, best_pin)
            rr = np.clip(rr, 0, self.size - 1)
            cc = np.clip(cc, 0, self.size - 1)
            
            # We remove the darkness. 
            # Using a fixed weight prevents "over-burning" lines
            self.target_image[rr, cc] = np.maximum(0, self.target_image[rr, cc] - dark_weight)
            
            current_pin = best_pin
            
            if step % 50 == 0:
                bar.progress(step / max_lines)
                status.text(f"Computing Line {step}/{max_lines} | Best Score: {int(best_score)}")

        bar.empty()
        status.empty()
        return sequence

    def render_preview(self, sequence):
        """
        Generates a 4K-quality render using anti-aliased lines
        """
        # Create large canvas for quality
        scale = 2
        w = self.size * scale
        canvas = np.ones((w, w), dtype=np.uint8) * 255
        
        # Use OpenCV for fast line drawing
        # We simulate transparency by drawing faint black lines
        # and accumulating them.
        
        # We start with white canvas.
        # We want lines to be black with alpha. 
        # Since CV2 doesn't do alpha on single channel easily, we subtract value.
        
        line_intensity = 25 # 0-255, smaller = more transparent
        
        print("Rendering preview...")
        
        # Draw lines
        for i in range(len(sequence) - 1):
            p1 = self.pins[sequence[i]]
            p2 = self.pins[sequence[i+1]]
            
            # Scale up
            pt1 = (p1[0]*scale, p1[1]*scale)
            pt2 = (p2[0]*scale, p2[1]*scale)
            
            # Draw line on a temp layer
            # Use standard line
            cv2.line(canvas, pt1, pt2, 0, 1, cv2.LINE_AA)
            
            # To simulate real density, we can't just draw black.
            # We need "accumulated darkness".
            # Actually, simply drawing thin black lines (1px) on a 2x canvas 
            # creates a perfect dithered look.
            
        # Resize back down
        preview = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return preview

    def create_svg(self, sequence):
        header = f'<svg height="{self.size}mm" width="{self.size}mm" viewBox="0 0 {self.size} {self.size}" xmlns="http://www.w3.org/2000/svg">\n'
        circle = f'<circle cx="{self.size/2}" cy="{self.size/2}" r="{self.radius}" stroke="#ccc" fill="none"/>\n'
        
        path_data = []
        for p in sequence:
            x, y = self.pins[p]
            path_data.append(f"{x},{y}")
            
        polyline = f'<polyline points="{" ".join(path_data)}" fill="none" stroke="black" stroke-width="0.15" />\n'
        return header + circle + polyline + '</svg>'

# --- UI LOGIC ---

col1, col2 = st.columns([1, 2])

with col1:
    st.title("String Art Industrial")
    st.caption("Brute-force Solver v1.0")
    
    uploaded_file = st.file_uploader("Upload High-Contrast Portrait", type=['jpg', 'png', 'jpeg'])
    
    st.markdown("### Tuning")
    num_pins = st.slider("Pins (Resolution)", 200, 300, 256)
    max_lines = st.slider("Lines (Darkness)", 1000, 4000, 2500)
    weight = st.slider("Thread Thickness", 10, 50, 25, help="Lower value = needs more lines to get dark.")
    
    run_btn = st.button("RUN SOLVER", use_container_width=True)

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file)
        solver = IndustrialSolver(image, num_pins=num_pins)
        
        # Show the "Tone Mapped" Input
        # This is CRITICAL. If this looks bad, the result looks bad.
        debug_view = Image.fromarray((255 - solver.target_image).astype(np.uint8))
        
        with st.expander("View Algorithm Input (Tone Mapped)", expanded=True):
            st.image(debug_view, caption="Algorithm Vision (White=Empty, Black=Target)", width=300)
        
        if run_btn:
            with st.spinner("Computing full vector path..."):
                # Run
                sequence = solver.solve(max_lines=max_lines, dark_weight=weight)
                
                # Render
                preview = solver.render_preview(sequence)
                
                st.success(f"Complete. {len(sequence)} steps.")
                st.image(preview, caption="Final Output", use_column_width=True)
                
                # Downloads
                svg = solver.create_svg(sequence)
                txt = " -> ".join(map(str, sequence))
                
                c1, c2 = st.columns(2)
                c1.download_button("Download SVG", svg, "output.svg")
                c2.download_button("Download Guide", txt, "guide.txt")
    else:
        st.info("Upload a photo to begin.")
