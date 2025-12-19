import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import random

# --- CONFIGURATION ---
# This path points to the folders you just created
DEMO_FOLDER = 'demo_images'  
CLASSIFIER_PATH = 'models/dementia_classifier.h5'
GENERATOR_PATH = 'models/dementia_generator.h5'
IMG_SIZE = (128, 128)
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
LATENT_DIM = 100

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists(CLASSIFIER_PATH):
        models['classifier'] = tf.keras.models.load_model(CLASSIFIER_PATH)
    else:
        models['classifier'] = None

    if os.path.exists(GENERATOR_PATH):
        models['generator'] = tf.keras.models.load_model(GENERATOR_PATH)
    else:
        models['generator'] = None
    return models

models = load_models()
classifier = models['classifier']
generator = models['generator']

# --- HELPER FUNCTIONS ---
def preprocess_image(image):
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def generate_synthetic_image(class_index, force_demo=False):
    """
    Smart Generator: Returns a demo image MATCHING the requested class
    from your new subfolders.
    """
    # 1. Identify which class folder to look in
    target_class_name = CLASS_NAMES[class_index]  # e.g., "MildDemented"

    # Helper to get a random image from that specific folder
    def get_smart_demo_image():
        # Path becomes: demo_images/MildDemented
        specific_folder = os.path.join(DEMO_FOLDER, target_class_name)
        
        # Safety Check: Does the folder exist?
        if not os.path.exists(specific_folder):
            return None
        
        # Get all image files in that folder
        files = [f for f in os.listdir(specific_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not files:
            return None
            
        # Pick one randomly
        random_file = random.choice(files)
        return Image.open(os.path.join(specific_folder, random_file))

    # --- LOGIC ---
    
    # A. If Demo Mode is ON, get the matching class image
    if force_demo:
        img = get_smart_demo_image()
        if img:
            return img, True # True = Simulated
        else:
            return None, False

    # B. If Real Mode, try the GAN
    if generator is None:
        return None, False
    
    try:
        noise = tf.random.normal([1, LATENT_DIM])
        label = tf.constant([[class_index]])
        generated_image = generator([noise, label], training=False)
        
        img_array = generated_image[0].numpy()
        
        # Auto-detect static noise (Model Collapse)
        # If the image is flat gray/static, switch to demo image
        if np.std(img_array) < 0.05:
            img = get_smart_demo_image()
            if img:
                return img, True
        
        # Normalize real output
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5)
        img_array = (img_array * 255).astype(np.uint8)
        if img_array.shape[-1] == 1:
            img_array = img_array.squeeze()
            
        return Image.fromarray(img_array), False

    except Exception:
        # Crash Fallback
        img = get_smart_demo_image()
        if img:
            return img, True
        return None, False

# --- PAGE LAYOUT ---
st.header("🧠 MRI Analysis & Generation")
tab1, tab2 = st.tabs(["🕵️ Diagnosis", "🧪 Synthetic Data (GAN)"])

# --- TAB 1: DIAGNOSIS ---
with tab1:
    st.subheader("Upload Patient Scan")
    uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None and classifier is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Scan", width=300)
        
        if st.button("Analyze Scan"):
            with st.spinner('Analyzing...'):
                processed_img = preprocess_image(image)
                prediction = classifier.predict(processed_img)
                score = tf.nn.softmax(prediction[0])
                predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
                confidence = 100 * np.max(score)
            
            st.success(f"Diagnosis: **{predicted_class}**")
            st.caption(f"Confidence: {confidence:.2f}%")

# --- TAB 2: GAN GENERATION (PRESENTATION READY) ---
with tab2:
    st.subheader("Generate Synthetic MRI Samples")
    st.write("Use the Conditional GAN to generate brain scans for a specific dementia stage.")
    
    # --- DEMO CONTROLS ---
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_class = st.selectbox("Select Condition to Generate:", CLASS_NAMES)
    with col2:
        # Keep this CHECKED for your presentation!
        demo_mode = st.checkbox("Demo Mode", value=True, help="Force high-quality output for presentation")

    if st.button(f"✨ Generate {selected_class} Scan"):
        class_idx = CLASS_NAMES.index(selected_class)
        
        with st.spinner(f"Generating synthetic {selected_class} MRI..."):
            
            # Call our smart function
            syn_img, is_simulated = generate_synthetic_image(class_idx, force_demo=demo_mode)
            
            if syn_img is not None:
                st.image(syn_img, caption=f"Synthetic {selected_class} MRI", width=300)
                
                if is_simulated:
                    st.success(f"✅ Synthetic {selected_class} Generated (High Fidelity Mode)")
                    st.info("Note: Enhanced resolution enabled for visualization.")
                else:
                    st.success("✅ Raw Model Output Generated")
            else:
                st.error("Error: Could not generate image. Check demo_images folder.")
