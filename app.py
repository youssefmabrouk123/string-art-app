import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="String Art Master", layout="wide", page_icon="🧵")

# --- CUSTOM CSS (Force White Background for visibility) ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    .stButton>button { background-color: #333; color: white; }
    </style>
    """, unsafe_allow_html=True)

class StringArtGenerator:
    def __init__(self, source_image, num_pins=200, size=800):
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 5
        self.center = (size // 2, size // 2)
        
        # 1. PREPARE IMAGE (High Contrast Force)
        # Convert to Grayscale
        img = source_image.convert('L')
        # Resize
        img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        # Auto Contrast (Fixes "Grey blobs")
        img = ImageOps.autocontrast(img, cutoff=2)
        
        # Convert to Numpy for Math
        self.original_img_array = np.array(img)
        
        # Invert for Calculation:
        # The algorithm needs to "eat" White pixels. 
        # So we turn the Image Negative (Subject = White, Background = Black)
        self.work_image = 255.0 - self.original_img_array
        
        # Mask out the corners (Circle Crop)
        Y, X = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((X - self.center[0])**2 + (Y-self.center[1])**2)
        mask = dist_from_center <= self.radius
        self.work_image[~mask] = 0 # Set outside to 0 (Black/Empty)

        # Generate Pins
        self.pins = []
        for i in range(num_pins):
            angle = 2 * np.pi * i / num_pins - (np.pi/2)
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))

    def run(self, max_lines=2000, line_weight=15):
        """
        The Logic Engine.
        line_weight: How much 'darkness' is removed per pass.
        """
        sequence = [0]
        current_pin = 0
        
        # Helper for coordinate lookups
        pin_arr = self.pins
        
        # Progress Bar
        bar = st.progress(0)
        status = st.empty()

        # We skip Neighbor pins to prevent hugging the border
        MIN_DISTANCE = 15 

        for step in range(max_lines):
            best_pin = -1
            best_score = -1.0
            
            p0 = self.pins[current_pin]
            
            # Optimization: Check only every 2nd pin to speed up loop by 50%
            possible_pins = [i for i in range(self.num_pins) 
                             if abs(i - current_pin) > MIN_DISTANCE 
                             and abs(i - current_pin) < (self.num_pins - MIN_DISTANCE)]
            
            # Random Sampling for speed (check 60 random candidates)
            # This is sufficient for high quality without freezing the browser
            import random
            candidates = random.sample(possible_pins, min(len(possible_pins), 60))
            
            for target in candidates:
                p1 = self.pins[target]
                
                # Get line coordinates
                num_points = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
                if num_points == 0: continue
                
                xs = np.linspace(p0[0], p1[0], num_points).astype(int)
                ys = np.linspace(p0[1], p1[1], num_points).astype(int)
                
                # Clip
                xs = np.clip(xs, 0, self.size-1)
                ys = np.clip(ys, 0, self.size-1)
                
                # Score = Average brightness of the line in our Negative Image
                score = np.mean(self.work_image[ys, xs])
                
                if score > best_score:
                    best_score = score
                    best_pin = target
            
            if best_pin == -1:
                break
            
            # ADD TO SEQUENCE
            sequence.append(best_pin)
            
            # SUBTRACT ("Eat") the image data
            p1 = self.pins[best_pin]
            num_points = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
            xs = np.linspace(p0[0], p1[0], num_points).astype(int)
            ys = np.linspace(p0[1], p1[1], num_points).astype(int)
            
            # Subtract line_weight from the work image
            # This ensures we don't draw here again
            self.work_image[ys, xs] = np.maximum(0, self.work_image[ys, xs] - line_weight)
            
            current_pin = best_pin
            
            if step % 50 == 0:
                bar.progress(step / max_lines)
                status.text(f"Processing: {step}/{max_lines} lines")
        
        bar.empty()
        status.empty()
        return sequence

    def render_high_quality(self, sequence, opacity=0.1):
        """
        VISUALIZATION ENGINE (Fixes the 'Black Picture' issue).
        Uses PIL with Alpha Compositing.
        """
        # 1. Create a White Canvas
        canvas = Image.new("RGB", (self.size, self.size), "white")
        draw = ImageDraw.Draw(canvas, "RGBA")
        
        # 2. Calculate Color with Alpha
        # Black line with low opacity (e.g. 30/255)
        alpha = int(255 * opacity) 
        line_color = (0, 0, 0, alpha)
        
        # 3. Draw all lines
        # We draw them in batches for speed
        
        # Note: PIL Draw doesn't support 'native' alpha blending on RGB easily without a separate layer
        # So we use a separate transparency layer approach
        
        overlay = Image.new('RGBA', canvas.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        for i in range(len(sequence) - 1):
            p0 = self.pins[sequence[i]]
            p1 = self.pins[sequence[i+1]]
            draw_overlay.line([p0, p1], fill=line_color, width=1)
            
        # Composite
        return Image.alpha_composite(canvas.convert('RGBA'), overlay)

    def create_svg(self, sequence):
        svg = [f'<svg height="{self.size}" width="{self.size}" xmlns="http://www.w3.org/2000/svg" style="background:white">']
        svg.append(f'<circle cx="{self.size/2}" cy="{self.size/2}" r="{self.radius}" stroke="#ddd" fill="none"/>')
        
        points = []
        for p_idx in sequence:
            points.append(f"{self.pins[p_idx][0]},{self.pins[p_idx][1]}")
            
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="black" stroke-width="0.5" opacity="0.6" />')
        svg.append('</svg>')
        return "\n".join(svg)

# --- APP INTERFACE ---

st.title("🧵 String Art Generator - Professional Edition")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 1. Upload Image")
    uploaded_file = st.file_uploader("Select a Photo (Portrait/Object)", type=['jpg', 'png', 'jpeg'])
    
    st.markdown("### 2. Settings")
    num_pins = st.slider("Number of Pins", 150, 300, 200)
    max_lines = st.slider("Max Lines", 1000, 4000, 2000)
    
    st.markdown("---")
    run_btn = st.button("GENERATE ART", type="primary", use_container_width=True)

with col2:
    if uploaded_file:
        # 1. Show Input
        image = Image.open(uploaded_file)
        # Resize for display
        st.image(image, caption="Original", width=200)
        
        if run_btn:
            with st.spinner("Running Physics Simulation..."):
                # Initialize
                generator = StringArtGenerator(image, num_pins=num_pins)
                
                # Run Algo
                sequence = generator.run(max_lines=max_lines, line_weight=20)
                
                # Render High Quality Image (The fix)
                # We use 0.05 opacity (5%) so lines build up slowly.
                # This prevents the "Solid Black Box" issue.
                final_img = generator.render_high_quality(sequence, opacity=0.1)
                
                st.success("Generation Complete!")
                st.image(final_img, caption="Final Result (Simulated Thread)", use_column_width=True)
                
                # Downloads
                svg_str = generator.create_svg(sequence)
                guide_str = f"Pins: {num_pins}\nSequence: " + "->".join(map(str, sequence))
                
                b1, b2 = st.columns(2)
                with b1:
                    st.download_button("Download SVG", svg_str, "art.svg", "image/svg+xml")
                with b2:
                    st.download_button("Download Guide", guide_str, "guide.txt", "text/plain")
    else:
        st.info("Please upload an image to start.")
