import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Pro String Art Studio", layout="wide", page_icon="🧵")

st.markdown("""
<style>
    .stApp { background-color: #111; color: #eee; }
    .stButton>button { background-color: #00cc66; color: white; font-weight: bold; }
    /* Force images to display nicely */
    img { border: 1px solid #333; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- ADVANCED IMAGE PROCESSING ENGINE ---

class StringArtProcessor:
    def __init__(self, pil_image, num_pins=200, size=800):
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 10
        self.center = (size // 2, size // 2)
        
        # 1. HIGH END PRE-PROCESSING
        # Convert to grayscale
        img = pil_image.convert('L')
        
        # Resize with high-quality resampling
        img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        
        # Convert to Numpy
        self.img_array = np.array(img)
        
        # 2. GENERATE PINS
        self.pins = []
        for i in range(num_pins):
            angle = 2 * np.pi * i / num_pins - (np.pi/2)
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))
            
    def apply_filters(self, contrast_boost=1.5, brightness_offset=0):
        """
        Prepares the image for the algorithm.
        This is where 'Bad Results' are fixed.
        """
        # Normalize to 0-1
        img_float = self.img_array.astype(float) / 255.0
        
        # Apply Sigmoid Contrast (S-Curve) to separate blacks and whites
        # This removes grey muddiness
        img_float = (img_float - 0.5) * contrast_boost + 0.5 + (brightness_offset/255.0)
        img_float = np.clip(img_float, 0, 1)
        
        # Convert back to 0-255
        processed = (img_float * 255).astype(np.uint8)
        
        # Apply Circular Mask (Force edges to White)
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        
        # Invert mask logic: We want background to be WHITE (255) so algo ignores it
        # The algorithm looks for DARK pixels (low values)
        # So we set outside of circle to 255
        final_img = np.ones_like(processed) * 255
        # Copy circle area
        cv2.bitwise_and(processed, processed, final_img, mask=mask)
        # Fix outside area (bitwise_and makes it black, we want white)
        final_img[mask == 0] = 255
        
        self.processed_img = final_img
        return final_img

    def solve(self, max_lines=2500, line_opacity=20):
        """
        The Solver.
        img: Input image (0=Black, 255=White)
        """
        # We work on an INVERTED copy. 
        # Algorithm wants to maximize sum. 
        # Standard Image: Black=0. 
        # Error Matrix: Black needs to be HIGH number (255).
        error_matrix = 255.0 - self.processed_img.astype(np.float32)
        
        # Clip negative values just in case
        error_matrix = np.clip(error_matrix, 0, 255)

        sequence = [0]
        current_pin = 0
        
        # Pre-calculate Pin Coordinates Array for speed
        pin_coords = np.array(self.pins)
        
        # Canvas for Real-time preview (float for precision)
        # Start with WHITE canvas (255)
        preview = np.ones((self.size, self.size), dtype=np.float32) * 255.0

        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        for step in range(max_lines):
            p0 = self.pins[current_pin]
            
            best_pin = -1
            max_score = -1.0
            
            # OPTIMIZATION: Only check pins that are far enough away
            # This prevents short ugly lines on the border
            min_dist_idx = int(self.num_pins * 0.15) # e.g. must skip 30 pins
            
            # Create search ranges
            # We want to scan the opposite side of the circle
            valid_indices = []
            total_pins = self.num_pins
            
            # Simple approach: Scan all pins except neighbors
            # To be fast: scan every 2nd or 3rd pin
            scan_step = 2 
            
            for i in range(0, total_pins, scan_step):
                dist = abs(i - current_pin)
                if dist > min_dist_idx and dist < (total_pins - min_dist_idx):
                    valid_indices.append(i)
            
            # Find Best Line
            for t in valid_indices:
                p1 = self.pins[t]
                
                # BRESENHAM / LINE ITERATOR
                # Get points along line
                # Fast numpy generation
                dist_px = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
                if dist_px == 0: continue
                
                xs = np.linspace(p0[0], p1[0], dist_px).astype(int)
                ys = np.linspace(p0[1], p1[1], dist_px).astype(int)
                
                # Clip to bounds
                xs = np.clip(xs, 0, self.size-1)
                ys = np.clip(ys, 0, self.size-1)
                
                # SCORE: Sum of brightness in Error Matrix
                # High value in Error Matrix = Dark spot in original image
                line_sum = np.sum(error_matrix[ys, xs])
                
                # We prefer long lines generally, so we don't divide strictly by length
                # or we use a slight power. Simple average works best for detailed faces.
                score = line_sum / dist_px
                
                if score > max_score:
                    max_score = score
                    best_pin = t
            
            if best_pin == -1:
                break
                
            # --- APPLY ---
            sequence.append(best_pin)
            
            # 1. Reduce Error (Subtract from Input)
            p1 = self.pins[best_pin]
            dist_px = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
            xs = np.linspace(p0[0], p1[0], dist_px).astype(int)
            ys = np.linspace(p0[1], p1[1], dist_px).astype(int)
            
            # Remove the "thread" from the error matrix
            # We subtract 'line_opacity'. 
            error_matrix[ys, xs] = np.maximum(0, error_matrix[ys, xs] - line_opacity)
            
            current_pin = best_pin
            
            # UI Updates
            if step % 100 == 0:
                progress_bar.progress(step / max_lines)
                status_text.text(f"Computing Line {step}/{max_lines}...")

        progress_bar.empty()
        status_text.empty()
        return sequence

    def render_realistic(self, sequence):
        """
        Generates a PNG that looks like REAL THREAD.
        Using PIL for high quality anti-aliasing.
        """
        # High res canvas (2x size for anti-aliasing)
        render_size = self.size * 2
        canvas = Image.new('RGB', (render_size, render_size), 'white')
        draw = ImageDraw.Draw(canvas, 'RGBA')
        
        # Scale pins
        scaled_pins = [(x*2, y*2) for x, y in self.pins]
        
        # Thread style
        # Black with very low alpha (transparency)
        # This is critical: 2000 lines at 100% opacity = black blob.
        # 2000 lines at 10% opacity = beautiful shading.
        thread_color = (0, 0, 0, 30) # Alpha 30/255 (approx 12%)
        
        # Draw frame
        cx, cy = self.center[0]*2, self.center[1]*2
        rad = self.radius * 2
        draw.ellipse((cx-rad, cy-rad, cx+rad, cy+rad), outline=(200,200,200), width=2)
        
        # Draw strings
        for i in range(len(sequence)-1):
            p0 = scaled_pins[sequence[i]]
            p1 = scaled_pins[sequence[i+1]]
            draw.line([p0, p1], fill=thread_color, width=2)
            
        # Downscale for crisp look
        canvas = canvas.resize((self.size, self.size), Image.Resampling.LANCZOS)
        return canvas
    
    def create_svg(self, sequence):
        svg = [f'<svg height="{self.size}mm" width="{self.size}mm" viewBox="0 0 {self.size} {self.size}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<circle cx="{self.size/2}" cy="{self.size/2}" r="{self.radius}" stroke="#ccc" fill="white"/>')
        
        points = []
        for p in sequence:
            points.append(f"{self.pins[p][0]},{self.pins[p][1]}")
            
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="black" stroke-width="0.2" opacity="0.8" />')
        svg.append('</svg>')
        return "\n".join(svg)

# --- UI ---

st.title("🧵 PRO String Art Generator")
st.caption("Algorithme V4: Contrast Boosting & Realistic Physics")

col_input, col_preview, col_result = st.columns([1, 1, 2])

with col_input:
    st.subheader("1. Input")
    uploaded_file = st.file_uploader("Upload Photo (Portrait)", type=['jpg', 'png'])
    
    st.markdown("---")
    st.subheader("2. Settings")
    
    # FILTER SETTINGS
    contrast = st.slider("Contrast Boost", 0.5, 3.0, 1.5, help="Higher = Remove grey tones. Essential for good results.")
    brightness = st.slider("Brightness", -100, 100, 0, help="Adjust if image is too dark/bright")
    
    # STRING SETTINGS
    num_pins = st.number_input("Pins", 150, 360, 200)
    max_lines = st.number_input("Lines", 1000, 5000, 2500)
    
    run_btn = st.button("🚀 GENERATE", use_container_width=True)

if uploaded_file:
    # Load
    image = Image.open(uploaded_file)
    processor = StringArtProcessor(image, num_pins=num_pins)
    
    # LIVE PREVIEW OF FILTERS
    # This is crucial so the user knows what the input looks like
    processed_preview = processor.apply_filters(contrast, brightness)
    
    with col_preview:
        st.subheader("2. Computer Vision")
        st.image(image, caption="Original", use_column_width=True)
        st.image(processed_preview, caption="Algorithm Input (What the AI sees)", use_column_width=True)
        st.info("👆 The B/W image must look clear for the result to be good.")

    if run_btn:
        with col_result:
            st.subheader("3. Final Render")
            with st.spinner("Calculating physics..."):
                # Solve
                seq = processor.solve(max_lines=max_lines)
                
                # Render
                final_img = processor.render_realistic(seq)
                
                st.image(final_img, caption="Simulated Result", use_column_width=True)
                
                # Downloads
                svg = processor.create_svg(seq)
                txt = f"Pins: {num_pins}\nLines: {len(seq)}\nSequence: " + "->".join(map(str, seq))
                
                c1, c2 = st.columns(2)
                c1.download_button("Download Vector (SVG)", svg, "string_art.svg", "image/svg+xml")
                c2.download_button("Download Guide (TXT)", txt, "guide.txt", "text/plain")
else:
    with col_result:
        st.info("Waiting for image...")
