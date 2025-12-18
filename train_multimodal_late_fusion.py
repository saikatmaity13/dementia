## ---------------------------------
##  FILE: train_multimodal_late_fusion.py
## ---------------------------------
#
# This script increases accuracy using "LATE FUSION".
#
# Strategy:
# 1. Train a Text Model (Logistic Regression) separately.
# 2. Train an Audio Model (Random Forest) separately.
# 3. Combine their predictions using Weighted Voting.
#    (We give Text more weight because we know it's stronger).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# --- CONFIGURATION ---
TEXT_FILE = "transcripts_with_ids.csv"
ACOUSTIC_FILE = "acoustic_features.csv"
PARALINGUISTIC_FILE = "paralinguistic_features.csv"
RANDOM_STATE = 42

print("--- Step 1: Loading & Merging Data ---")

# 1. Load Text
if not os.path.exists(TEXT_FILE):
    print(f"Error: '{TEXT_FILE}' not found. Run 'clean_data_v2.py' first.")
    sys.exit()
df_text = pd.read_csv(TEXT_FILE)

# 2. Load Audio
if not os.path.exists(ACOUSTIC_FILE):
    print(f"Error: '{ACOUSTIC_FILE}' not found.")
    sys.exit()
df_acoustic = pd.read_csv(ACOUSTIC_FILE)

if os.path.exists(PARALINGUISTIC_FILE):
    df_para = pd.read_csv(PARALINGUISTIC_FILE)
    df_audio = pd.merge(df_acoustic, df_para, on=['File_Name', 'Diagnosis', 'Full_Path'], how='inner')
    # Cleanup columns
    cols_to_drop = [c for c in df_audio.columns if '_y' in c]
    df_audio = df_audio.drop(columns=cols_to_drop)
    df_audio.columns = df_audio.columns.str.replace('_x', '')
else:
    df_audio = df_acoustic

# 3. Merge All
df_final = pd.merge(df_text, df_audio, on=['File_Name', 'Diagnosis'], how='inner')
df_final = df_final.dropna(subset=['Processed_Text'])
# Identify audio columns (everything that is not text/metadata)
metadata_cols = ['Diagnosis', 'Processed_Text', 'File_Name', 'Full_Path', 'Transcript']
audio_cols = [c for c in df_final.columns if c not in metadata_cols]
# Fill audio NaNs with 0
df_final[audio_cols] = df_final[audio_cols].fillna(0)

print(f"Final Dataset: {len(df_final)} samples")

# --- Step 2: Setup Models ---
print("\n--- Step 2: Building Ensemble Model ---")

X = df_final # Contains both text and audio columns
y_raw = df_final['Diagnosis']

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# --- Define the Text "Expert" ---
# Uses TF-IDF + Logistic Regression
text_pipeline = Pipeline([
    ('selector', ColumnTransformer([
        ('text', TfidfVectorizer(max_features=5000, min_df=3, max_df=0.9), 'Processed_Text')
    ])),
    ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=RANDOM_STATE))
])

# --- Define the Audio "Expert" ---
# Uses Scaling + Random Forest (RF is better for audio features)
audio_pipeline = Pipeline([
    ('selector', ColumnTransformer([
        ('audio', StandardScaler(), audio_cols)
    ])),
    ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE))
])

# --- Combine with Soft Voting ---
# weights=[2, 1] means Text vote counts 2x as much as Audio vote
ensemble_model = VotingClassifier(
    estimators=[
        ('text_expert', text_pipeline),
        ('audio_expert', audio_pipeline)
    ],
    voting='soft',
    weights=[2, 1] 
)

# --- Step 3: Train & Evaluate ---
print("\n--- Step 3: Training & Evaluating ---")
ensemble_model.fit(X_train, y_train)

y_pred = ensemble_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print("  LATE FUSION RESULTS")
print("="*30)
print(f"Test Accuracy: {acc:.4f}")
print("(This combines the strengths of both models!)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))