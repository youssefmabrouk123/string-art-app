import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from skimage.draw import line
import math
import io

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="String Art Generator Pro",
    page_icon="🧵",
    layout="wide"
)

# --- LOGIQUE ALGORITHMIQUE (Cachée du client) ---
def process_image(image_input, num_pins, max_lines, line_weight):
    # 1. Préparation de l'image
    img = Image.open(image_input).convert('L') # Niveaux de gris
    
    # Redimensionnement carré pour le calcul (600x600)
    size = 600
    img = ImageOps.fit(img, (size, size), Image.LANCZOS)
    
    # Conversion en array numpy inversé (255 = blanc, 0 = noir)
    # L'algo cherche le noir, donc on inverse : le noir devient des valeurs élevées
    img_array = 255 - np.array(img)
    current_img = img_array.copy()

    # 2. Placement des clous (Cercle)
    pins = []
    radius = (size / 2) - 5
    center = size / 2
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        x = int(center + radius * math.cos(angle))
        y = int(center + radius * math.sin(angle))
        pins.append((x, y))
    
    # 3. Algorithme Glouton (Calcul du fil)
    sequence = [0] # On commence au clou 0
    current_pin = 0
    pin_coords = np.array(pins)
    
    # Barre de progression visuelle
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(max_lines):
        best_pin = -1
        max_darkness = -1
        
        # Mise à jour barre de progression tous les 10%
        if step % (max_lines // 10) == 0:
            prog = int((step / max_lines) * 100)
            progress_bar.progress(prog)
            status_text.text(f"Calcul du fil : {step}/{max_lines} lignes...")

        # On saute le clou actuel et le précédent pour éviter les aller-retours immédiats
        prev_pin = sequence[-2] if len(sequence) > 1 else -1
        
        # Optimisation: on ne vérifie pas TOUS les clous, mais un échantillon ou tous sauf voisins
        # Ici on vérifie tout pour la qualité
        for target_pin in range(num_pins):
            if target_pin == current_pin or target_pin == prev_pin:
                continue
                
            # Distance minimale pour éviter les fils trop courts sur les bords
            if abs(target_pin - current_pin) < 5: 
                 continue

            start_x, start_y = pins[current_pin]
            end_x, end_y = pins[target_pin]
            
            rr, cc = line(start_x, start_y, end_x, end_y)
            
            # Score : somme de la noirceur sous la ligne
            line_score = np.sum(current_img[rr, cc])
            
            if line_score > max_darkness:
                max_darkness = line_score
                best_pin = target_pin
        
        if best_pin == -1:
            break
            
        sequence.append(best_pin)
        
        # Soustraire la ligne de l'image
        s_x, s_y = pins[current_pin]
        e_x, e_y = pins[best_pin]
        rr, cc = line(s_x, s_y, e_x, e_y)
        current_img[rr, cc] = np.maximum(0, current_img[rr, cc] - line_weight)
        
        current_pin = best_pin

    progress_bar.progress(100)
    status_text.text("Terminé !")
    return pins, sequence

def create_svg(pins, sequence):
    """Génère le code SVG."""
    size = 600
    svg = [f'<svg height="{size}" width="{size}" xmlns="http://www.w3.org/2000/svg" style="background: white;">']
    
    # Calque Clous (pour perçage)
    svg.append('<g id="nails" opacity="0.3">')
    for idx, (x, y) in enumerate(pins):
        svg.append(f'<circle cx="{x}" cy="{y}" r="2" fill="red" />')
        if idx % 10 == 0: # Numéroter tous les 10 clous
             svg.append(f'<text x="{x+5}" y="{y+5}" font-family="Arial" font-size="10" fill="red">{idx}</text>')
    svg.append('</g>')

    # Calque Fil
    points_str = " ".join([f"{pins[p][0]},{pins[p][1]}" for p in sequence])
    svg.append(f'<polyline points="{points_str}" style="fill:none;stroke:black;stroke-width:0.5;opacity:0.7" />')
    
    svg.append('</svg>')
    return "\n".join(svg)

def create_guide(sequence):
    """Génère le guide texte."""
    txt = "GUIDE PAS-À-PAS - STRING ART\n"
    txt += f"Nombre total de passages : {len(sequence)}\n"
    txt += "------------------------------------------------\n\n"
    
    for i in range(0, len(sequence), 10):
        chunk = sequence[i:i+10]
        arrow_chain = " → ".join(map(str, chunk))
        txt += f"Étape {i+1} : {arrow_chain}\n"
    
    return txt

# --- INTERFACE UTILISATEUR (Frontend) ---

# En-tête
st.title("🧵 String Art Generator")
st.markdown("Transformez vos photos en **œuvres d'art filaire**. Téléchargez une image, ajustez les paramètres et obtenez votre plan de fabrication.")

# Colonnes pour la mise en page
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Configuration")
    uploaded_file = st.file_uploader("Choisissez une image", type=['jpg', 'png', 'jpeg'])
    
    st.subheader("Paramètres")
    num_pins = st.slider("Nombre de Clous", 100, 360, 200, step=10, help="Plus il y a de clous, plus le détail est fin sur les bords.")
    max_lines = st.slider("Densité (Nombre de lignes)", 500, 4000, 2000, step=100, help="Le nombre total de passages du fil.")
    line_weight = st.slider("Épaisseur du fil (Contraste)", 10, 100, 25, help="Valeur élevée = fil épais (l'image s'assombrit vite). Valeur faible = fil fin.")
    
    generate_btn = st.button("🚀 Générer la simulation", type="primary", use_container_width=True)

with col2:
    st.header("2. Résultat")
    
    if uploaded_file is not None:
        # Afficher l'image originale
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", width=300)
        
        if generate_btn:
            with st.spinner("L'artiste numérique travaille... (Cela peut prendre 15-30 secondes)"):
                # Lancer le calcul
                pins, sequence = process_image(uploaded_file, num_pins, max_lines, line_weight)
                
                # Générer les sorties
                svg_data = create_svg(pins, sequence)
                txt_data = create_guide(sequence)
                
                # Afficher le résultat SVG via HTML
                st.success("Génération réussie !")
                st.markdown(f'<div style="border:1px solid #ccc; border-radius:5px; padding:10px;">{svg_data}</div>', unsafe_allow_html=True)
                
                st.divider()
                
                # Zone de Téléchargement
                st.subheader("3. Télécharger les fichiers")
                d_col1, d_col2 = st.columns(2)
                
                with d_col1:
                    st.download_button(
                        label="📥 Télécharger le Plan (SVG)",
                        data=svg_data,
                        file_name="string_art_plan.svg",
                        mime="image/svg+xml"
                    )
                
                with d_col2:
                    st.download_button(
                        label="📄 Télécharger le Guide (TXT)",
                        data=txt_data,
                        file_name="guide_montage.txt",
                        mime="text/plain"
                    )
    else:
        st.info("👈 Commencez par télécharger une image dans la colonne de gauche.")

# Footer
st.markdown("---")
st.markdown("*Outil développé pour les artisans DIY.*")
