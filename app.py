import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import math
import io

st.set_page_config(page_title="String Art Pro Studio", layout="wide", page_icon="🧵")

# --- STYLE CSS POUR UN LOOK PRO ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #1E1E1E;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    .stProgress > div > div > div > div {
        background-color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

def preprocess_image_pro(image_pil, size=800):
    """
    Professional preprocessing: High contrast & Circular Mask
    """
    # 1. Convert to grayscale
    img = image_pil.convert('L')
    
    # 2. Resize high quality
    img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
    
    # 3. Auto-Contrast (This fixes the "Nothing appears" issue)
    img = ImageOps.autocontrast(img, cutoff=2)
    
    # 4. Apply Circular Mask (so the corners don't distract algorithm)
    # Create a mask
    mask = Image.new('L', (size, size), 0)
    draw = Image.new('L', (size, size), 0)
    # Draw a filled white circle
    import PIL.ImageDraw as ImageDraw
    d = ImageDraw.Draw(mask)
    d.ellipse((10, 10, size-10, size-10), fill=255)
    
    # Apply mask: outside circle becomes White (255)
    img_arr = np.array(img)
    mask_arr = np.array(mask)
    img_arr[mask_arr == 0] = 255 
    
    return img_arr

def generate_string_art(img_array, num_pins, max_lines, opacity_weight):
    """
    Optimized Core Algorithm using OpenCV for speed
    """
    height, width = img_array.shape
    center = (width // 2, height // 2)
    radius = min(center) - 5
    
    # 1. Define Pins
    pins = []
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        pins.append((x, y))
    
    # 2. Invert image for calculation (Algorithm seeks Brightness, so we invert Input)
    # We want the thread (black) to go where the image is black.
    # So we make the image: High Value = Dark, Low Value = Light
    work_img = 255 - img_array
    work_img = work_img.astype(np.float32)

    sequence = [0]
    current_pin = 0
    
    # Canvas for the REALISTIC PREVIEW (White background)
    preview_canvas = np.ones((height, width), dtype=np.uint8) * 255
    
    # Status updates
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Pre-calculate Pin Coordinates for speed
    pin_coords = np.array(pins)
    
    # --- THE LOOP ---
    for step in range(max_lines):
        best_pin = -1
        max_score = -1.0
        
        # Logic: Look at a subset of pins to speed up (optional) or all pins
        # We skip neighbors to ensure lines cross the center
        start_p = (current_pin + 10) % num_pins
        
        # We iterate through possible targets
        # NOTE: Using a simple heuristic here for stability
        best_pin = -1
        best_score = -1
        
        # Get coordinates of current pin
        p0 = pins[current_pin]
        
        # To optimize: In a real PRO app, we use C++ or numba. 
        # Here we use a simplified scan for Streamlit Cloud limits.
        
        # Iterate over a range of pins (skipping neighbors)
        targets = [i for i in range(num_pins) if abs(i - current_pin) > 10]
        
        # Let's check 50 random pins instead of ALL pins to prevent TimeOut on Cloud
        # (This makes it faster and still looks random/artistic)
        import random
        targets_sample = random.sample(targets, min(len(targets), 60)) 

        for t in targets_sample:
            p1 = pins[t]
            
            # Line Iterator (Bresenham or similar via CV2)
            # We sample the line intensity
            # cv2.lineIterator is not directly exposed in python easily for sum, 
            # so we use createLineIterator logic or simple profile
            
            # Quick hack: measure line intensity
            # Generate x,y coordinates along the line
            num_points = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
            x_vals = np.linspace(p0[0], p1[0], num_points).astype(int)
            y_vals = np.linspace(p0[1], p1[1], num_points).astype(int)
            
            # Ensure within bounds
            x_vals = np.clip(x_vals, 0, width-1)
            y_vals = np.clip(y_vals, 0, height-1)
            
            # Calculate average intensity
            score = np.sum(work_img[y_vals, x_vals])
            
            if score > best_score:
                best_score = score
                best_pin = t

        if best_pin == -1:
            break
            
        # ADD LINE
        sequence.append(best_pin)
        
        # SUBTRACT from Work Image (Don't go there again)
        p1 = pins[best_pin]
        cv2.line(work_img, p0, p1, (0), thickness=1) # Set to 0 (black/empty)
        
        # DRAW on Preview (Realistic Thread)
        # We draw a black line with transparency. Since we can't do alpha in 1-channel easily in cv2 loop:
        # We simply darken the pixel.
        cv2.line(preview_canvas, p0, p1, (0), thickness=1, lineType=cv2.LINE_AA)

        current_pin = best_pin
        
        if step % 50 == 0:
            progress_bar.progress(min(step / max_lines, 1.0))
            status_text.caption(f"Tissage du fil : Ligne {step}")

    progress_bar.empty()
    status_text.empty()
    
    return pins, sequence, preview_canvas

def create_svg(pins, sequence, size=800):
    svg = [f'<svg height="{size}" width="{size}" xmlns="http://www.w3.org/2000/svg" style="background: white;">']
    
    # Circle border
    svg.append(f'<circle cx="{size/2}" cy="{size/2}" r="{(size/2)-5}" stroke="black" stroke-width="1" fill="none"/>')
    
    # Thread
    path_data = []
    for p_idx in sequence:
        x, y = pins[p_idx]
        path_data.append(f"{x},{y}")
    
    # Polyline is much lighter for browser to render than individual lines
    points = " ".join(path_data)
    svg.append(f'<polyline points="{points}" style="fill:none;stroke:black;stroke-width:0.5;opacity:0.8" />')
    
    # Pin Numbers (Optional - can clog the SVG)
    svg.append(f'<g id="numbers" opacity="0">') # Hidden by default
    for i, (x,y) in enumerate(pins):
        if i % 10 == 0:
             svg.append(f'<text x="{x}" y="{y}" font-size="10">{i}</text>')
    svg.append(f'</g>')
    
    svg.append('</svg>')
    return "\n".join(svg)

# --- UI LAYOUT ---

st.title("🧵 String Art Generator Pro")
st.caption("Algorithme de rendu vectoriel haute performance")

col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    st.info("1. Importez votre image")
    uploaded_file = st.file_uploader("Fichier image (JPG, PNG)", type=['jpg', 'png', 'jpeg'])
    
    st.write("---")
    st.info("2. Réglages Techniques")
    
    num_pins = st.number_input("Nombre de Clous", min_value=100, max_value=360, value=200, step=10)
    max_lines = st.slider("Densité de fil (Lignes)", 1000, 4000, 2500)
    
    st.write("---")
    run_btn = st.button("GÉNÉRER LE RENDU", type="primary")

with col2:
    if uploaded_file:
        # Load and Show Input
        image_pil = Image.open(uploaded_file)
        
        # Preprocess
        with st.spinner("Optimisation du contraste..."):
            processed_img = preprocess_image_pro(image_pil)
            
        # Show the "Computer Vision" view (what the algo sees)
        st.image(processed_img, caption="Vue Algorithmique (Contraste forcé)", width=200)

        if run_btn:
            with st.spinner("Calcul de la trajectoire optimale..."):
                pins, sequence, preview_arr = generate_string_art(
                    processed_img, 
                    num_pins=num_pins, 
                    max_lines=max_lines, 
                    opacity_weight=20
                )
                
                # Show Result
                st.success(f"Génération terminée ! {len(sequence)} lignes tracées.")
                
                # Display the realistic preview
                st.image(preview_arr, caption="Simulation Réaliste", use_column_width=True)
                
                # Prepare Downloads
                svg_str = create_svg(pins, sequence)
                guide_str = f"GUIDE DE MONTAGE\nNombre de clous: {num_pins}\n\nSéquence:\n" + " -> ".join(map(str, sequence))
                
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("⬇️ Télécharger SVG (Vecteur)", svg_str, "projet_art.svg", "image/svg+xml")
                with c2:
                    st.download_button("⬇️ Télécharger Guide (Txt)", guide_str, "instructions.txt", "text/plain")

    else:
        # Placeholder for professional look
        st.warning("En attente d'image...")
        st.markdown("""
        <div style="border: 2px dashed #ccc; padding: 50px; text-align: center; border-radius: 10px;">
            📂 Glissez votre image ici pour commencer
        </div>
        """, unsafe_allow_html=True)
