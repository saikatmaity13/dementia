## ---------------------------------
##  FILE: train_multimodal.py
## ---------------------------------
#
# This script trains a model on BOTH Text and Audio features.
# It uses "Early Fusion":
# 1. Converts Text to Numbers (TF-IDF)
# 2. Scales Audio Features (StandardScaler)
# 3. Concatenates them side-by-side
# 4. Trains a LogisticRegression model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import sys
import os

# --- CONFIGURATION ---
TEXT_FILE = "transcripts_with_ids.csv"
ACOUSTIC_FILE = "acoustic_features.csv"
PARALINGUISTIC_FILE = "paralinguistic_features.csv"
RANDOM_STATE = 42

print("--- Step 1: Loading Data ---")

# 1. Load Text
if not os.path.exists(TEXT_FILE):
    print(f"Error: '{TEXT_FILE}' not found. Run 'clean_data_v2.py' first.")
    sys.exit()
df_text = pd.read_csv(TEXT_FILE)
print(f"Loaded Text: {len(df_text)} records")

# 2. Load Audio
if not os.path.exists(ACOUSTIC_FILE):
    print(f"Error: '{ACOUSTIC_FILE}' not found.")
    sys.exit()
df_acoustic = pd.read_csv(ACOUSTIC_FILE)

if os.path.exists(PARALINGUISTIC_FILE):
    df_para = pd.read_csv(PARALINGUISTIC_FILE)
    # Merge Acoustic + Para
    df_audio = pd.merge(df_acoustic, df_para, on=['File_Name', 'Diagnosis', 'Full_Path'], how='inner')
    
    # Drop duplicate duration columns if they exist (e.g. duration_x, duration_y)
    # We keep one and drop the others
    cols_to_drop = [c for c in df_audio.columns if '_y' in c]
    df_audio = df_audio.drop(columns=cols_to_drop)
    # Rename _x columns back to normal
    df_audio.columns = df_audio.columns.str.replace('_x', '')
    
    print(f"Loaded & Merged Audio: {len(df_audio)} records")
else:
    df_audio = df_acoustic
    print(f"Loaded Acoustic Only: {len(df_audio)} records")

# --- Step 2: Merge Text and Audio ---
print("\n--- Step 2: Merging Text & Audio ---")

# Merge on File_Name and Diagnosis
df_final = pd.merge(df_text, df_audio, on=['File_Name', 'Diagnosis'], how='inner')

# Drop missing values
df_final = df_final.dropna(subset=['Processed_Text'])
# Fill any audio NaNs with 0
audio_cols = [c for c in df_audio.columns if c not in ['File_Name', 'Diagnosis', 'Full_Path']]
df_final[audio_cols] = df_final[audio_cols].fillna(0)

print(f"Final Combined Dataset: {len(df_final)} samples")

# --- Step 3: Prepare Features ---
print("\n--- Step 3: Feature Engineering ---")

y = df_final['Diagnosis']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Split Data First (Stratified)
X_train_df, X_test_df, y_train, y_test = train_test_split(
    df_final, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded
)

print(f"Train Set: {len(X_train_df)}")
print(f"Test Set:  {len(X_test_df)}")

# A. Process Text (TF-IDF)
print("  Vectorizing Text...")
tfidf = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.9)
X_train_text = tfidf.fit_transform(X_train_df['Processed_Text'])
X_test_text = tfidf.transform(X_test_df['Processed_Text'])

# B. Process Audio (Scaling)
print("  Scaling Audio...")
scaler = StandardScaler()
X_train_audio = scaler.fit_transform(X_train_df[audio_cols])
X_test_audio = scaler.transform(X_test_df[audio_cols])

# C. Concatenate
print("  Combining Features...")
X_train_combined = hstack([X_train_text, X_train_audio])
X_test_combined = hstack([X_test_text, X_test_audio])

print(f"  Total Features: {X_train_combined.shape[1]}")

# --- Step 4: Train & Evaluate ---
print("\n--- Step 4: Training Fusion Model ---")

model = LogisticRegression(
    class_weight='balanced',
    solver='liblinear',
    random_state=RANDOM_STATE,
    C=1.0
)

model.fit(X_train_combined, y_train)

print("\n" + "="*30)
print("  MULTIMODAL MODEL RESULTS")
print("="*30)

y_pred = model.predict(X_test_combined)
acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))