import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import random

# --- CONFIGURATION ---
DATA_DIR = 'reference_images'
CLASSIFIER_PATH = 'models/dementia_classifier.h5'
GENERATOR_PATH = 'models/dementia_generator.h5'
IMG_SIZE = (128, 128)
# CLASS NAMES MUST MATCH TRAINING ORDER EXACTLY
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
LATENT_DIM = 100

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists(CLASSIFIER_PATH):
        models['classifier'] = tf.keras.models.load_model(CLASSIFIER_PATH)
    else:
        st.error(f"⚠️ Classifier not found at {CLASSIFIER_PATH}")
        models['classifier'] = None

    if os.path.exists(GENERATOR_PATH):
        models['generator'] = tf.keras.models.load_model(GENERATOR_PATH)
    else:
        st.warning(f"⚠️ Generator not found at {GENERATOR_PATH}")
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

# --- FIXED GENERATOR FUNCTION ---
def generate_synthetic_image(class_index):
    """Generates an image using Noise + Class Label"""
    if generator is None:
        return None
    
    # 1. Create Random Noise
    noise = tf.random.normal([1, LATENT_DIM])
    
    # 2. Create Label Input (The specific class we want)
    label = tf.constant([[class_index]])  # Shape (1, 1)
    
    # 3. Generate Image (Pass BOTH inputs as a list)
    generated_image = generator([noise, label], training=False)
    
    # 4. Denormalize
    # Assuming output is [-1, 1] -> [0, 255]
    generated_image = (generated_image[0, :, :, 0] * 127.5 + 127.5).numpy().astype(np.uint8)
    
    return generated_image

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

# --- TAB 2: GAN GENERATION (UPDATED) ---
with tab2:
    st.subheader("Generate Synthetic MRI Samples")
    st.write("Use the Conditional GAN to generate brain scans for a specific dementia stage.")
    
    if generator is not None:
        # User selects the class they want to generate
        selected_class = st.selectbox("Select Condition to Generate:", CLASS_NAMES)
        
        if st.button(f"✨ Generate {selected_class} Scan"):
            # Get the index (0, 1, 2, or 3)
            class_idx = CLASS_NAMES.index(selected_class)
            
            with st.spinner(f"Generating synthetic {selected_class} MRI..."):
                syn_img = generate_synthetic_image(class_idx)
                
                if syn_img is not None:
                    st.image(syn_img, caption=f"Synthetic {selected_class} MRI", width=300)
                    st.success("Image generated successfully!")
                else:
                    st.error("Error generating image.")
    else:
        st.warning("Generator model not loaded.")
