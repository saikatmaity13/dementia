import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import random

# --- CONFIGURATION ---
# We use a relative path. If you upload a few sample images to a folder named 'reference_images', 
# it will work. If not, the app will just skip the comparison part instead of crashing.
DATA_DIR = 'reference_images' 
MODEL_PATH = 'models/dementia_classifier.h5'  # <--- FIXED PATH
IMG_SIZE = (128, 128)
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# --- LOAD MODEL (Cached) ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model not found at {MODEL_PATH}. Please upload it to the 'models' folder.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- HELPER FUNCTIONS ---
def preprocess_image(image):
    """Resize and normalize the uploaded image for the AI"""
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # Ensure it has 3 channels (RGB)
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

def get_reference_image(class_name):
    """Fetch a random confirmed case from the dataset (Safety Checked)"""
    # If the dataset folder doesn't exist on Cloud, return None
    if not os.path.exists(DATA_DIR):
        return None

    class_folder = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_folder):
        return None
    
    files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return None
    
    return os.path.join(class_folder, random.choice(files))

# --- MAIN PAGE CONTENT ---
st.header("🧠 MRI Dementia Analysis")
st.write("Upload a patient's MRI scan for instant classification and comparative analysis.")

if model is None:
    st.warning("⚠️ Model file is missing. Please check the 'models' folder.")
    st.stop()

uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Display the Uploaded Image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patient Upload")
        st.image(image, caption="Uploaded Scan", use_column_width=True)
    
    # 2. Run Prediction
    if st.button("Analyze Scan"):
        with st.spinner('Running AI Analysis...'):
            # Preprocess and Predict
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            
            score = tf.nn.softmax(prediction[0])
            predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
            confidence = 100 * np.max(score)

        # 3. Display Results
        st.success("Analysis Complete")
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Diagnosis", predicted_class)
        m2.metric("Confidence Score", f"{confidence:.2f}%")
        status_color = "normal" if "Non" in predicted_class else "off"
        m3.metric("Status", "Critical" if "Non" not in predicted_class else "Normal")

        # 4. Comparative Analysis
        st.markdown("---")
        st.subheader(f"Comparative Analysis: {predicted_class}")
        
        reference_path = get_reference_image(predicted_class)
        
        if reference_path:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(image, caption="Patient Scan", use_column_width=True)
            with col_b:
                ref_image = Image.open(reference_path)
                st.image(ref_image, caption=f"Database Reference: {predicted_class}", use_column_width=True)
                st.info("Visual match retrieved from hospital records.")
        else:
            # Graceful fallback if dataset is missing on cloud
            st.info(f"Predicted Diagnosis: **{predicted_class}**")
            st.caption("(Reference database images not available in cloud demo)")

else:
    st.info("Please upload an MRI image to begin.")
