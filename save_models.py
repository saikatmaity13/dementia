## ---------------------------------
##  FILE: save_models.py
## ---------------------------------
#
# Run this ONCE to save your trained models and vectorizers
# so the Streamlit app can load them quickly.

import pandas as pd
import numpy as np
import os
import joblib # For saving models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- CONFIGURATION ---
TEXT_FILE = "transcripts_with_ids.csv"
ACOUSTIC_FILE = "acoustic_features.csv"
PARALINGUISTIC_FILE = "paralinguistic_features.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

print("--- Step 1: Loading Data ---")
try:
    df_text = pd.read_csv(TEXT_FILE)
    df_acoustic = pd.read_csv(ACOUSTIC_FILE)
except FileNotFoundError as e:
    print(f"Error: Missing file. {e}")
    print("Make sure you run 'clean_data_v2.py' and 'extract_acoustic_features.py' first.")
    exit()

# Prepare Audio Data
if os.path.exists(PARALINGUISTIC_FILE):
    df_para = pd.read_csv(PARALINGUISTIC_FILE)
    df_audio = pd.merge(df_acoustic, df_para, on=['File_Name', 'Diagnosis', 'Full_Path'], how='inner')
    # Cleanup columns
    cols_to_drop = [c for c in df_audio.columns if '_y' in c]
    df_audio = df_audio.drop(columns=cols_to_drop)
    df_audio.columns = df_audio.columns.str.replace('_x', '')
else:
    df_audio = df_acoustic

# Merge All
df_final = pd.merge(df_text, df_audio, on=['File_Name', 'Diagnosis'], how='inner')
df_final = df_final.dropna(subset=['Processed_Text'])
metadata_cols = ['Diagnosis', 'Processed_Text', 'File_Name', 'Full_Path', 'Transcript']
audio_cols = [c for c in df_final.columns if c not in metadata_cols]
df_final[audio_cols] = df_final[audio_cols].fillna(0)

print(f"Training on {len(df_final)} samples...")

y = df_final['Diagnosis']

# --- Step 2: Save TEXT Model ---
print("Training & Saving Text Model...")
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, min_df=3, max_df=0.9)),
    ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=RANDOM_STATE))
])
text_pipeline.fit(df_final['Processed_Text'], y)
joblib.dump(text_pipeline, os.path.join(MODEL_DIR, 'text_model.pkl'))

# --- Step 3: Save AUDIO Model ---
print("Training & Saving Audio Model...")
audio_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE))
])
audio_pipeline.fit(df_final[audio_cols], y)
joblib.dump(audio_pipeline, os.path.join(MODEL_DIR, 'audio_model.pkl'))

# --- Step 4: Save Feature List ---
# We need to know which audio columns to expect in the app
joblib.dump(audio_cols, os.path.join(MODEL_DIR, 'audio_features.pkl'))

print("\n--- Success! Models saved to 'models/' folder ---")