import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from skimage.draw import line
import time

st.set_page_config(page_title="String Art Safe Mode", page_icon="🧵")

class StringArtGenerator:
    def __init__(self, num_pins=200, size=500):
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 5
        self.center = (size // 2, size // 2)
        
        # Generate Pin Coordinates (cached)
        self.pins = []
        for i in range(num_pins):
            angle = 2 * np.pi * i / num_pins - np.pi/2
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((y, x)) # Store as (Row, Col) for numpy

    def preprocess(self, img_pil, contrast=1.0, brightness=0, invert=False):
        # 1. Grayscale & Resize
        img = img_pil.convert('L')
        img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # 2. Numpy conversion
        arr = np.array(img, dtype=np.float32)
        
        # 3. Contrast/Brightness Math
        # (Value - 128) * Contrast + 128 + Brightness
        arr = (arr - 128) * contrast + 128 + brightness
        arr = np.clip(arr, 0, 255)
        
        # 4. Mask Circle
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - self.center[1])**2 + (Y - self.center[0])**2)
        mask = dist > self.radius
        arr[mask] = 255 # White background outside
        
        # 5. Target Calculation
        # If invert=False: We want BLACK thread on WHITE background.
        # So we look for DARK pixels (Low values).
        # Target Matrix: High value = Needs Thread.
        if invert:
            # Use this if image is "White text on Black background"
            self.target = arr 
        else:
            # Standard Photo
            self.target = 255 - arr
            
        self.target[mask] = 0 # Ignore outside circle
        
        return self.target

    def solve(self, max_lines=2000, line_weight=20):
        sequence = [0]
        current_pin = 0
        
        # Working matrix (mutable)
        error = self.target.copy()
        
        # Progress bar
        bar = st.progress(0)
        status = st.empty()
        
        start_time = time.time()
        
        # Skip neighbors check
        skip = 15
        
        for step in range(max_lines):
            best_pin = -1
            best_score = -1.0
            
            # Get current pin coordinates
            r0, c0 = self.pins[current_pin]
            
            # Define candidates (simple loop)
            # We check every 2nd pin to speed it up without losing much quality
            candidates = [p for p in range(0, self.num_pins, 2) 
                          if abs(p - current_pin) > skip 
                          and abs(p - current_pin) < (self.num_pins - skip)]
            
            for pin_idx in candidates:
                r1, c1 = self.pins[pin_idx]
                
                # Generate line pixels using Scikit-Image (Very robust)
                rr, cc = line(r0, c0, r1, c1)
                
                # Check bounds
                if len(rr) == 0: continue
                
                # Score: Average intensity
                line_vals = error[rr, cc]
                score = np.mean(line_vals)
                
                if score > best_score:
                    best_score = score
                    best_pin = pin_idx
            
            if best_pin == -1:
                break
                
            # Commit
            sequence.append(best_pin)
            
            # Subtract
            r1, c1 = self.pins[best_pin]
            rr, cc = line(r0, c0, r1, c1)
            
            # "Draw" the thread by removing value from error matrix
            error[rr, cc] = np.maximum(0, error[rr, cc] - line_weight)
            
            current_pin = best_pin
            
            if step % 50 == 0:
                bar.progress(min(step/max_lines, 1.0))
                status.text(f"Lines: {step}/{max_lines}")
        
        bar.empty()
        status.empty()
        return sequence

    def render(self, sequence, opacity=20):
        # Render with PIL
        scale = 2
        w = self.size * scale
        img = Image.new("RGBA", (w, w), (255, 255, 255, 255))
        draw = Image.new("RGBA", (w, w), (0,0,0,0))
        d = ImageDraw.Draw(draw)
        
        alpha = int((opacity / 100) * 255)
        color = (0, 0, 0, alpha)
        
        # Scale pins
        pins_scaled = [(c*scale, r*scale) for (r,c) in self.pins]
        
        for i in range(len(sequence)-1):
            p0 = pins_scaled[sequence[i]]
            p1 = pins_scaled[sequence[i+1]]
            d.line([p0, p1], fill=color, width=2)
            
        # Composite
        img = Image.alpha_composite(img, draw)
        return img.resize((self.size, self.size), Image.Resampling.LANCZOS)

# --- UI ---
st.title("🧵 String Art Generator")
st.markdown("**Safe Mode:** Optimized for reliability.")

col1, col2 = st.columns([1, 2])

with col1:
    f = st.file_uploader("Upload Photo")
    
    st.divider()
    st.subheader("Settings")
    n_pins = st.slider("Pins", 100, 300, 200)
    n_lines = st.slider("Max Lines", 1000, 4000, 2000)
    weight = st.slider("Darkness Weight", 10, 100, 25, help="Lower = Better shading. Higher = Blacker lines.")
    
    st.divider()
    invert = st.checkbox("Invert Colors?", False, help="Use this if your result is a negative image.")
    
    run = st.button("Run", type="primary")

with col2:
    if f:
        img = Image.open(f)
        
        gen = StringArtGenerator(num_pins=n_pins)
        target = gen.preprocess(img, invert=invert, contrast=1.2)
        
        # Show what the computer sees
        st.image(255-target, caption="Computer Vision (White = Empty, Black = Thread)", width=200)
        
        if run:
            seq = gen.solve(n_lines, weight)
            final = gen.render(seq, opacity=15)
            
            st.success(f"Done! {len(seq)} lines.")
            st.image(final, use_column_width=True)
            
            # SVG
            svg_pts = " ".join([f"{gen.pins[p][1]},{gen.pins[p][0]}" for p in seq])
            svg = f'<svg height="500" width="500"><polyline points="{svg_pts}" fill="none" stroke="black" stroke-width="0.1" opacity="0.5"/></svg>'
            st.download_button("Download SVG", svg, "art.svg")
