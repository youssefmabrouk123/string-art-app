import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import time

st.set_page_config(page_title="Pro String Art Studio V5", layout="wide", page_icon="🧵")

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; color: #eee; }
    .stButton>button { background-color: #00cc66; color: white; font-weight: bold; border-radius: 8px; }
    img { border: 1px solid #333; border-radius: 8px; }
    .success-box { padding: 1rem; background: #1a3a1a; border-radius: 8px; border: 1px solid #00cc66; }
</style>
""", unsafe_allow_html=True)

class StringArtProcessor:
    def __init__(self, pil_image, num_pins=200, size=800):
        self.num_pins = num_pins
        self.size = size
        self.radius = (size // 2) - 5
        self.center = (size // 2, size // 2)
        self.original_image = pil_image
        self.line_cache = {}
        
    def preprocess_image(self, clahe_clip=2.0, clahe_grid=8, gamma=1.0, 
                         blur_size=1, invert=False, edge_enhance=0.0):
        """Advanced preprocessing for optimal string art results"""
        # Convert to grayscale
        img = self.original_image.convert('L')
        
        # Resize with high-quality resampling - fit to circle
        img = ImageOps.fit(img, (self.size, self.size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)
        
        # Apply Gaussian blur to reduce noise
        if blur_size > 0:
            blur_k = blur_size * 2 + 1
            img_array = cv2.GaussianBlur(img_array, (blur_k, blur_k), 0)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is KEY for face details - enhances local contrast
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        img_array = clahe.apply(img_array)
        
        # Gamma correction (< 1 = brighter midtones, > 1 = darker midtones)
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                              for i in np.arange(256)]).astype("uint8")
            img_array = cv2.LUT(img_array, table)
        
        # Edge enhancement (optional - helps define features)
        if edge_enhance > 0:
            edges = cv2.Canny(img_array, 50, 150)
            edges = cv2.GaussianBlur(edges, (3, 3), 0)
            img_array = cv2.addWeighted(img_array, 1.0, edges, -edge_enhance * 0.3, 0)
        
        # Invert if needed (dark thread on light background vs light thread on dark)
        if invert:
            img_array = 255 - img_array
        
        # Apply circular mask - CRITICAL: make outside WHITE (255)
        # The algorithm seeks dark pixels, so white = ignored
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        
        # Create output with white background
        result = np.ones((self.size, self.size), dtype=np.uint8) * 255
        result[mask == 255] = img_array[mask == 255]
        
        self.processed_img = result
        return result
    
    def generate_pins(self):
        """Generate evenly spaced pins around the circle"""
        self.pins = []
        for i in range(self.num_pins):
            angle = 2 * np.pi * i / self.num_pins - (np.pi / 2)
            x = int(self.center[0] + self.radius * np.cos(angle))
            y = int(self.center[1] + self.radius * np.sin(angle))
            self.pins.append((x, y))
        return self.pins
    
    def get_line_pixels(self, pin_a, pin_b):
        """Get all pixels along a line using Bresenham - with caching"""
        cache_key = (min(pin_a, pin_b), max(pin_a, pin_b))
        if cache_key in self.line_cache:
            return self.line_cache[cache_key]
        
        p0, p1 = self.pins[pin_a], self.pins[pin_b]
        
        # Bresenham's line algorithm for accurate pixel coverage
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
                pixels.append((y0, x0))  # Note: (row, col) for numpy
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        self.line_cache[cache_key] = pixels
        return pixels
    
    def solve(self, max_lines=3000, line_weight=20, min_distance=0.1, 
              early_stop_threshold=0.5, progress_callback=None):
        """
        Main solving algorithm - greedy approach with squared error minimization
        
        Key insight: We want to minimize (target - current)^2
        Adding a line reduces error if the line passes through dark areas
        """
        self.generate_pins()
        
        # Pre-cache all valid lines
        min_pin_dist = int(self.num_pins * min_distance)
        
        # Target image: INVERT so dark=high value (what we want to cover)
        # 0 = white (ignore), 255 = black (cover with strings)
        target = (255.0 - self.processed_img.astype(np.float32))
        
        # Error/residual matrix - starts as target, we subtract as we add strings
        residual = target.copy()
        
        # Track our result (starts white = 255, gets darker)
        canvas = np.ones((self.size, self.size), dtype=np.float32) * 255.0
        
        sequence = [0]  # Start at pin 0
        current_pin = 0
        
        # For early stopping
        prev_total_error = np.sum(residual ** 2)
        stall_count = 0
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        for step in range(max_lines):
            best_pin = -1
            best_score = -np.inf
            
            # Check all valid destination pins
            for candidate in range(self.num_pins):
                # Skip pins too close (creates short ugly lines)
                dist = abs(candidate - current_pin)
                dist = min(dist, self.num_pins - dist)  # Wrap around
                
                if dist < min_pin_dist:
                    continue
                
                # Get line pixels
                pixels = self.get_line_pixels(current_pin, candidate)
                if len(pixels) < 10:
                    continue
                
                # Calculate score: sum of residual values along line
                # Higher residual = darker area that needs coverage
                score = sum(residual[p[0], p[1]] for p in pixels)
                
                # Normalize by length (prefer longer lines that cover more)
                score = score / np.sqrt(len(pixels))
                
                if score > best_score:
                    best_score = score
                    best_pin = candidate
            
            if best_pin == -1 or best_score <= 0:
                status.text(f"Stopping early at line {step} - no improvement possible")
                break
            
            # Apply the selected line
            sequence.append(best_pin)
            pixels = self.get_line_pixels(current_pin, best_pin)
            
            # Update residual: subtract the "thread" contribution
            for py, px in pixels:
                residual[py, px] = max(0, residual[py, px] - line_weight)
                canvas[py, px] = max(0, canvas[py, px] - line_weight * 0.3)
            
            current_pin = best_pin
            
            # Progress updates
            if step % 50 == 0:
                progress_bar.progress(min(step / max_lines, 1.0))
                current_error = np.sum(residual ** 2)
                improvement = (prev_total_error - current_error) / prev_total_error * 100
                status.text(f"Line {step}/{max_lines} | Error reduction: {improvement:.1f}%")
                
                # Early stopping check
                if improvement < early_stop_threshold and step > 500:
                    stall_count += 1
                    if stall_count > 3:
                        status.text(f"Converged at line {step}")
                        break
                else:
                    stall_count = 0
                prev_total_error = current_error
        
        progress_bar.empty()
        status.empty()
        
        self.sequence = sequence
        self.canvas = canvas
        return sequence
    
    def render_preview(self, line_alpha=25):
        """Quick preview render"""
        canvas = Image.new('L', (self.size, self.size), 255)
        draw = ImageDraw.Draw(canvas, 'L')
        
        for i in range(len(self.sequence) - 1):
            p0 = self.pins[self.sequence[i]]
            p1 = self.pins[self.sequence[i + 1]]
            # Accumulate darkness
            draw.line([p0, p1], fill=max(0, 255 - line_alpha), width=1)
        
        return canvas
    
    def render_realistic(self, thread_alpha=18, thread_width=1):
        """High-quality realistic render with transparency blending"""
        # Use 2x size for anti-aliasing
        render_size = self.size * 2
        
        # RGBA canvas for proper alpha blending
        canvas = Image.new('RGBA', (render_size, render_size), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        
        # Scale pins
        scaled_pins = [(x * 2, y * 2) for x, y in self.pins]
        
        # Draw frame circle
        cx, cy = self.center[0] * 2, self.center[1] * 2
        rad = self.radius * 2
        draw.ellipse((cx - rad - 5, cy - rad - 5, cx + rad + 5, cy + rad + 5), 
                     outline=(180, 180, 180, 255), width=3)
        
        # Draw pins
        pin_size = 4
        for px, py in scaled_pins:
            draw.ellipse((px - pin_size, py - pin_size, px + pin_size, py + pin_size),
                        fill=(60, 60, 60, 255))
        
        # Draw strings with low alpha for realistic overlap
        thread_color = (0, 0, 0, thread_alpha)
        
        for i in range(len(self.sequence) - 1):
            p0 = scaled_pins[self.sequence[i]]
            p1 = scaled_pins[self.sequence[i + 1]]
            draw.line([p0, p1], fill=thread_color, width=thread_width * 2)
        
        # Downscale with anti-aliasing
        canvas = canvas.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Convert to RGB
        background = Image.new('RGB', canvas.size, (255, 255, 255))
        background.paste(canvas, mask=canvas.split()[3])
        
        return background
    
    def create_svg(self):
        """Create vector SVG for printing/cutting"""
        svg = [
            f'<svg width="{self.size}mm" height="{self.size}mm" '
            f'viewBox="0 0 {self.size} {self.size}" xmlns="http://www.w3.org/2000/svg">',
            f'<circle cx="{self.center[0]}" cy="{self.center[1]}" r="{self.radius}" '
            f'stroke="#ccc" stroke-width="2" fill="white"/>',
        ]
        
        # Draw pins
        for i, (px, py) in enumerate(self.pins):
            svg.append(f'<circle cx="{px}" cy="{py}" r="3" fill="#333" />')
            # Pin numbers (optional, for guide)
            if i % 10 == 0:
                svg.append(f'<text x="{px}" y="{py-8}" font-size="8" '
                          f'text-anchor="middle" fill="#666">{i}</text>')
        
        # Draw path
        if len(self.sequence) > 1:
            path_d = f'M {self.pins[self.sequence[0]][0]} {self.pins[self.sequence[0]][1]}'
            for idx in self.sequence[1:]:
                path_d += f' L {self.pins[idx][0]} {self.pins[idx][1]}'
            
            svg.append(f'<path d="{path_d}" fill="none" stroke="black" '
                      f'stroke-width="0.3" stroke-opacity="0.7"/>')
        
        svg.append('</svg>')
        return '\n'.join(svg)
    
    def create_instructions(self):
        """Create human-readable winding instructions"""
        lines = [
            "STRING ART WINDING GUIDE",
            "=" * 40,
            f"Total Pins: {self.num_pins}",
            f"Total Connections: {len(self.sequence) - 1}",
            f"Frame Diameter: {self.radius * 2} units",
            "",
            "PIN SEQUENCE:",
            "-" * 40,
        ]
        
        # Group into sets of 10 for readability
        for i in range(0, len(self.sequence), 10):
            chunk = self.sequence[i:i+10]
            lines.append(f"[{i+1:4d}] " + " → ".join(map(str, chunk)))
        
        return '\n'.join(lines)


# ============ STREAMLIT UI ============

st.title("🧵 String Art Generator Pro V5")
st.caption("Advanced algorithm with CLAHE preprocessing, Bresenham lines, and optimized greedy solver")

# Layout
col_upload, col_settings = st.columns([1, 1])
col_preview, col_result = st.columns([1, 1])

with col_upload:
    st.subheader("📷 Upload Image")
    uploaded = st.file_uploader("Choose a portrait photo", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Original", use_column_width=True)

with col_settings:
    st.subheader("⚙️ Settings")
    
    with st.expander("🎨 Image Preprocessing", expanded=True):
        clahe_clip = st.slider("CLAHE Contrast", 1.0, 4.0, 2.5, 
                               help="Adaptive contrast - higher = more detail")
        gamma = st.slider("Gamma", 0.5, 2.0, 1.0, 
                         help="< 1 = brighter, > 1 = darker")
        blur = st.slider("Noise Reduction", 0, 3, 1)
        invert = st.checkbox("Invert (for light thread on dark)", False)
    
    with st.expander("🔧 Algorithm Parameters", expanded=True):
        num_pins = st.slider("Number of Pins", 100, 360, 200)
        max_lines = st.slider("Maximum Lines", 1000, 5000, 3000)
        line_weight = st.slider("Line Weight", 10, 40, 22, 
                               help="How much each string 'darkens' the image")
        min_dist = st.slider("Min Pin Distance", 0.05, 0.25, 0.12,
                            help="Prevents short lines near edges")
    
    with st.expander("🖼️ Render Settings"):
        thread_alpha = st.slider("Thread Opacity", 10, 40, 18)
        render_size = st.selectbox("Output Size", [600, 800, 1000, 1200], index=1)

    generate_btn = st.button("🚀 GENERATE STRING ART", use_container_width=True, type="primary")

# Processing
if uploaded and generate_btn:
    with st.spinner("Processing..."):
        # Initialize processor
        processor = StringArtProcessor(image, num_pins=num_pins, size=render_size)
        
        # Preprocess
        with col_preview:
            st.subheader("🔬 Preprocessing")
            processed = processor.preprocess_image(
                clahe_clip=clahe_clip,
                gamma=gamma,
                blur_size=blur,
                invert=invert
            )
            st.image(processed, caption="Algorithm Input (what the solver sees)", 
                    use_column_width=True)
            st.info("👆 Dark areas = where strings will be placed")
        
        # Solve
        with col_result:
            st.subheader("🎨 Result")
            sequence = processor.solve(
                max_lines=max_lines,
                line_weight=line_weight,
                min_distance=min_dist
            )
            
            # Render
            result = processor.render_realistic(thread_alpha=thread_alpha)
            st.image(result, caption=f"String Art ({len(sequence)-1} connections)", 
                    use_column_width=True)
            
            # Downloads
            st.markdown("### 📥 Downloads")
            c1, c2, c3 = st.columns(3)
            
            # SVG
            svg_data = processor.create_svg()
            c1.download_button("📐 Vector (SVG)", svg_data, "string_art.svg", "image/svg+xml")
            
            # Instructions
            txt_data = processor.create_instructions()
            c2.download_button("📋 Guide (TXT)", txt_data, "winding_guide.txt", "text/plain")
            
            # PNG
            from io import BytesIO
            buf = BytesIO()
            result.save(buf, format='PNG', quality=95)
            c3.download_button("🖼️ Image (PNG)", buf.getvalue(), "string_art.png", "image/png")
            
            st.success(f"✅ Generated {len(sequence)-1} string connections across {num_pins} pins!")

elif not uploaded:
    with col_preview:
        st.info("👈 Upload a portrait photo to begin")
        st.markdown("""
        **Tips for best results:**
        - Use high-contrast portraits
        - Face should fill most of the frame
        - Good lighting with clear shadows
        - Avoid busy backgrounds
        """)
