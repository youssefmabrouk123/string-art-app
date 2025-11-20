import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import time

st.set_page_config(page_title="HD String Art Generator", layout="wide", page_icon="🧵")

# --- CSS FOR DARK MODE ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stButton>button { background-color: #007bff; color: white; font-weight: bold; border-radius: 5px; }
    .stImage { border: 1px solid #333; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

class HDStringArt:
    def __init__(self, pil_image, num_pins=250, size=1000): # Increased default resolution
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 5
        self.center = (size // 2, size // 2)
        
        # 1. HD PRE-PROCESSING
        img = pil_image.convert('L')
        img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        self.original_pil = img
        self.img_array = np.array(img)
        
        # Generate Pins
        self.pins = []
        for i in range(num_pins):
            # Subtract pi/2 to put Pin 0 at the top (12 o'clock)
            angle = 2 * np.pi * i / num_pins - (np.pi/2)
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))

    def enhance_image(self, contrast=1.2, brightness=10, sharpness=2.0):
        """
        CRITICAL STEP: Prepares the image specifically for thread rendering.
        """
        # 1. Sharpen (Unsharp Mask) - This fixes the "Blurry Face" issue
        # We apply it multiple times based on the slider
        img = self.original_pil
        for _ in range(int(sharpness)):
             img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        
        # 2. Convert to Numpy for Contrast Math
        arr = np.array(img).astype(float)
        
        # 3. Apply Contrast & Brightness
        # Formula: New = (Old - 128) * Contrast + 128 + Brightness
        arr = (arr - 128) * contrast + 128 + brightness
        
        # 4. Clip values
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        
        # 5. Circular Mask (Clean Edges)
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius - 2, 255, -1)
        
        # Apply mask: Outside becomes WHITE (255)
        final_img = np.ones_like(arr) * 255
        cv2.bitwise_and(arr, arr, final_img, mask=mask)
        final_img[mask == 0] = 255
        
        self.processed_img = final_img
        return final_img

    def compute_paths(self, max_lines=3000, min_distance=20):
        """
        The Solver with "Edge Cleaning" logic.
        min_distance: Prevents short lines that look like dirt.
        """
        # Invert image: We need "Darkness Map" (0=White/Empty, 255=Black/Target)
        error_matrix = 255.0 - self.processed_img.astype(np.float32)
        
        sequence = [0]
        current_pin = 0
        
        # Visualization Canvas
        preview_canvas = np.ones((self.size, self.size), dtype=np.uint8) * 255
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Loop
        for step in range(max_lines):
            best_pin = -1
            best_score = -1.0
            
            p0 = self.pins[current_pin]
            
            # --- SMART PIN SELECTION ---
            # We only check pins that are NOT neighbors.
            # This forces the lines to cut across the face.
            
            # Create a list of valid targets
            # Logic: Target must be at least 'min_distance' pins away
            valid_targets = []
            for i in range(self.num_pins):
                dist = abs(i - current_pin)
                # Handle circular wrapping (e.g. pin 0 and pin 200 are close)
                dist = min(dist, self.num_pins - dist)
                
                if dist > min_distance:
                    valid_targets.append(i)
            
            # Optimization: Check a random sample of valid targets (e.g. 100) to be fast
            import random
            if len(valid_targets) > 80:
                valid_targets = random.sample(valid_targets, 80)
            
            for t in valid_targets:
                p1 = self.pins[t]
                
                # Measure line darkness
                dist_px = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
                if dist_px == 0: continue
                
                xs = np.linspace(p0[0], p1[0], dist_px).astype(int)
                ys = np.linspace(p0[1], p1[1], dist_px).astype(int)
                
                # Bound check
                xs = np.clip(xs, 0, self.size-1)
                ys = np.clip(ys, 0, self.size-1)
                
                # Score = Sum of pixel darkness
                line_sum = np.sum(error_matrix[ys, xs])
                score = line_sum / dist_px
                
                if score > best_score:
                    best_score = score
                    best_pin = t
            
            if best_pin == -1:
                break
                
            sequence.append(best_pin)
            
            # Subtract (Standard Weight)
            p1 = self.pins[best_pin]
            dist_px = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
            xs = np.linspace(p0[0], p1[0], dist_px).astype(int)
            ys = np.linspace(p0[1], p1[1], dist_px).astype(int)
            
            # Weight = 30 (Hardcoded for balance)
            error_matrix[ys, xs] = np.maximum(0, error_matrix[ys, xs] - 30)
            
            current_pin = best_pin
            
            if step % 100 == 0:
                progress_bar.progress(step / max_lines)
                status_text.text(f"Weaving thread {step}/{max_lines}...")
                
        return sequence

    def render_hd(self, sequence, color=(0,0,0), opacity=0.15):
        """
        Render high-res preview
        """
        scale = 2 # Super-sampling
        w = self.size * scale
        canvas = Image.new('RGBA', (w, w), (255,255,255,255))
        draw = ImageDraw.Draw(canvas)
        
        # Draw Circle Frame
        cx, cy = self.center[0]*scale, self.center[1]*scale
        r = self.radius * scale
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=(200,200,200), width=3)
        
        # Calculate color with alpha
        r, g, b = color
        a = int(255 * opacity)
        ink = (r, g, b, a)
        
        # Scale Pins
        s_pins = [(x*scale, y*scale) for x,y in self.pins]
        
        # Draw Lines
        for i in range(len(sequence)-1):
            p0 = s_pins[sequence[i]]
            p1 = s_pins[sequence[i+1]]
            draw.line([p0, p1], fill=ink, width=2)
            
        # Resize down
        return canvas.resize((self.size, self.size), Image.Resampling.LANCZOS)
    
    def create_svg(self, sequence):
        svg = [f'<svg height="{self.size}mm" width="{self.size}mm" viewBox="0 0 {self.size} {self.size}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<circle cx="{self.size/2}" cy="{self.size/2}" r="{self.radius}" stroke="#ccc" fill="none"/>')
        path_data = " ".join([f"{self.pins[p][0]},{self.pins[p][1]}" for p in sequence])
        svg.append(f'<polyline points="{path_data}" fill="none" stroke="black" stroke-width="0.2" opacity="0.7" />')
        svg.append('</svg>')
        return "\n".join(svg)

# --- UI LAYOUT ---
st.title("🧵 String Art HD Generator")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Upload & Tune")
    uploaded = st.file_uploader("Upload Portrait", type=["jpg", "png", "jpeg"])
    
    if uploaded:
        st.markdown("### 🛠 Image Enhancer")
        st.info("Adjust these until the face looks EXTREMELY clear.")
        
        sharpness = st.slider("Sharpness (Definition)", 0.0, 5.0, 2.0, help="Higher = More detail in eyes/mouth")
        contrast = st.slider("Contrast", 0.5, 2.0, 1.3)
        brightness = st.slider("Brightness", -50, 50, 10)
        
        st.markdown("### ⚙️ Machine Settings")
        pins = st.slider("Nail Count", 200, 360, 280, help="More nails = Better resolution")
        lines = st.slider("Thread Density", 1500, 4000, 2500)
        min_dist = st.slider("Min Line Length", 0, 50, 20, help="Prevents messy short lines on edges")
        
        btn = st.button("GENERATE HD PREVIEW", type="primary", use_container_width=True)

with col2:
    if uploaded:
        img = Image.open(uploaded)
        engine = HDStringArt(img, num_pins=pins)
        
        # PREVIEW WINDOW
        enhanced = engine.enhance_image(contrast, brightness, sharpness)
        st.image(enhanced, caption="Step 1: How the Machine Sees Your Photo", width=300)
        
        if btn:
            with st.spinner("Calculating optimal thread path..."):
                # Run Algo
                seq = engine.compute_paths(max_lines=lines, min_distance=min_dist)
                
                # Render
                res = engine.render_hd(seq)
                st.image(res, caption="Step 2: Final Thread Simulation", use_column_width=True)
                
                # Download
                svg = engine.create_svg(seq)
                txt = f"Nails: {pins}\nLines: {len(seq)}\nSequence: " + "->".join(map(str, seq))
                
                c1, c2 = st.columns(2)
                c1.download_button("Download Vector (SVG)", svg, "art.svg", "image/svg+xml")
                c2.download_button("Download Steps (TXT)", txt, "guide.txt", "text/plain")
    else:
        st.markdown("## Instructions for Best Results:")
        st.markdown("""
        1. Upload a **Close-up Portrait** (Face only is best).
        2. Use the **Sharpness Slider** to make the eyes pop.
        3. Set **Nail Count** to at least 250.
        """)
