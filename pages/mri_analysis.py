import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import random
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = 'reference_images' 
CLASSIFIER_PATH = 'models/dementia_classifier.h5'
GENERATOR_PATH = 'models/dementia_generator.h5'  # <--- Path to Generator
IMG_SIZE = (128, 128)
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
LATENT_DIM = 100  # Standard noise dimension for GANs (Change if you used different dim)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    models = {}
    
    # Load Classifier
    if os.path.exists(CLASSIFIER_PATH):
        models['classifier'] = tf.keras.models.load_model(CLASSIFIER_PATH)
    else:
        st.error(f"⚠️ Classifier not found at {CLASSIFIER_PATH}")
        models['classifier'] = None

    # Load Generator
    if os.path.exists(GENERATOR_PATH):
        models['generator'] = tf.keras.models.load_model(GENERATOR_PATH)
    else:
        st.warning(f"⚠️ Generator not found at {GENERATOR_PATH}. Synthetic image feature will be disabled.")
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

def generate_synthetic_image():
    """Generates an image using the loaded GAN model"""
    if generator is None:
        return None
    
    # 1. Create random noise (Latent Vector)
    noise = tf.random.normal([1, LATENT_DIM])
    
    # 2. Generate Image
    generated_image = generator(noise, training=False)
    
    # 3. Denormalize (Scale from [-1, 1] to [0, 1] usually)
    # Adjust this math depending on how you trained your GAN!
    generated_image = (generated_image[0, :, :, 0] * 127.5 + 127.5).numpy().astype(np.uint8)
    
    return generated_image

# --- PAGE LAYOUT ---
st.header("🧠 MRI Analysis & Generation")

# TABS for different modes
tab1, tab2 = st.tabs(["🕵️ Diagnosis (Classifier)", "🧪 Synthetic Data (GAN)"])

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
            st.progress(int(confidence))
            st.caption(f"Confidence: {confidence:.2f}%")

# --- TAB 2: GAN GENERATION ---
with tab2:
    st.subheader("Generate Synthetic MRI Samples")
    st.write("Use the Generative Adversarial Network (GAN) to create synthetic brain scans for research.")
    
    if generator is not None:
        if st.button("✨ Generate New Brain Scan"):
            with st.spinner("Generating pixel data from noise..."):
                syn_img = generate_synthetic_image()
                
                if syn_img is not None:
                    st.image(syn_img, caption="AI-Generated Synthetic MRI", width=300)
                    st.success("Image generated successfully!")
                else:
                    st.error("Error generating image.")
    else:
        st.warning("Generator model not loaded. Please upload 'dementia_generator.h5' to the models folder.")
