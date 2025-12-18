## ---------------------------------
##  FILE: train_audio_model.py
## ---------------------------------
#
# This script trains a classifier using ONLY the audio features
# (Acoustic + Paralinguistic).
#
# It merges the two CSVs, handles missing data, and
# trains a RandomForest model.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import sys
import os

# --- 1. CONFIGURATION ---
ACOUSTIC_FILE = "acoustic_features.csv"
PARALINGUISTIC_FILE = "paralinguistic_features.csv"
RANDOM_STATE = 42

print("--- Step 1: Loading Feature Files ---")

# Load Acoustic Features
if not os.path.exists(ACOUSTIC_FILE):
    print(f"Error: '{ACOUSTIC_FILE}' not found. Run 'extract_acoustic_features.py' first.")
    sys.exit()
df_acoustic = pd.read_csv(ACOUSTIC_FILE)
print(f"Loaded {len(df_acoustic)} acoustic records.")

# Load Paralinguistic Features
if os.path.exists(PARALINGUISTIC_FILE):
    df_para = pd.read_csv(PARALINGUISTIC_FILE)
    print(f"Loaded {len(df_para)} paralinguistic records.")
    
    # Merge them on File_Name
    # We use 'inner' join to keep only files that have BOTH types of features
    df = pd.merge(df_acoustic, df_para, on=['File_Name', 'Diagnosis', 'Full_Path'], how='inner')
    print(f"Merged Data Size: {len(df)} records.")
else:
    print(f"Warning: '{PARALINGUISTIC_FILE}' not found. Training on Acoustic features only.")
    df = df_acoustic

# --- 2. PREPROCESSING ---
print("\n--- Step 2: Preprocessing ---")

# Drop non-numeric columns that aren't features
# (We keep Diagnosis as target)
cols_to_drop = ['File_Name', 'Full_Path', 'Diagnosis']
# Identify feature columns (everything else)
feature_cols = [c for c in df.columns if c not in cols_to_drop]

X = df[feature_cols]
y_raw = df['Diagnosis']

# Handle Missing Values (NaNs)
# Some files might be too short to calculate pitch, leading to NaNs
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encode Target
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

print(f"Features used: {len(feature_cols)}")
print(f"Target classes: {class_names}")

# Scale Features (Important for audio data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.20, 
    random_state=RANDOM_STATE, 
    stratify=y
)

# --- 4. TRAIN MODEL ---
print("\n--- Step 3: Training Random Forest Model ---")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)

model.fit(X_train, y_train)
print("Training complete.")

# --- 5. EVALUATE ---
print("\n" + "="*30)
print("  AUDIO MODEL RESULTS")
print("="*30)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# --- 6. FEATURE IMPORTANCE ---
print("\n--- Top 5 Most Important Audio Features ---")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(5):
    print(f"{i+1}. {feature_cols[indices[i]]} ({importances[indices[i]]:.4f})")