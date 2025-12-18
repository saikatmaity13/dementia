## ---------------------------------
##  FILE: train_on_synthetic_only.py
## ---------------------------------
#
# This script trains a model ONLY on the 100% synthetic data
# and then evaluates it on the 100% ORIGINAL data.
# This tests how well the synthetic data generalizes.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

# --- 1. Configuration ---
ORIGINAL_DATA_FILE = 'transcripts_classifier_clean.csv'
AUGMENTED_DATA_FILE = 'augmented_data/transcripts_augmented_mlm.csv'
RANDOM_STATE = 42

print(f"--- Step 1: Loading Data ---")
try:
    df_original = pd.read_csv(ORIGINAL_DATA_FILE)
    df_augmented_full = pd.read_csv(AUGMENTED_DATA_FILE)
except FileNotFoundError as e:
    print(f"Error: Could not find a file. {e}")
    print("Please make sure both 'transcripts_classifier_clean.csv' and")
    print("'augmented_data/transcripts_augmented_mlm.csv' exist.")
    sys.exit()

# Drop NaNs
df_original = df_original.dropna(subset=['Processed_Text', 'Diagnosis'])
df_augmented_full = df_augmented_full.dropna(subset=['Processed_Text', 'Diagnosis'])

print(f"Loaded {len(df_original)} original records.")
print(f"Loaded {len(df_augmented_full)} total augmented records (original + synthetic).")

# --- 2. Separate Synthetic Data for Training ---
print("\n--- Step 2: Separating Synthetic and Original Data ---")

# To find *only* the synthetic data, we can merge the two dataframes
# and find the rows that are *only* in the augmented file.
df_merged = pd.merge(
    df_augmented_full,
    df_original,
    on=['Processed_Text', 'Diagnosis'],
    how='left',
    indicator=True
)

# The synthetic samples will have '_merge' == 'left_only'
df_train_synthetic = df_merged[df_merged['_merge'] == 'left_only'].copy()

# The evaluation set is the original data
df_eval_original = df_original.copy()

if len(df_train_synthetic) == 0:
    print("\nError: No synthetic data was found.")
    print("This might happen if the 'Processed_Text' was modified")
    print("in the original file after augmentation.")
    sys.exit()

print(f"Found {len(df_train_synthetic)} synthetic samples for TRAINING.")
print(f"Found {len(df_eval_original)} original samples for EVALUATION.")

# --- 3. Define X and y for Train/Eval ---
X_train = df_train_synthetic['Processed_Text']
y_train_raw = df_train_synthetic['Diagnosis']

X_eval = df_eval_original['Processed_Text']
y_eval_raw = df_eval_original['Diagnosis']

# --- 4. Label Encoding ---
# We fit the encoder on the ORIGINAL data's labels
# to ensure consistent encoding
le = LabelEncoder()
le.fit(df_original['Diagnosis'])
class_names = le.classes_

y_train = le.transform(y_train_raw)
y_eval = le.transform(y_eval_raw)

print(f"\nLabels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- 5. Feature Extraction (TF-IDF) ---
print("\n--- Step 5: Feature Extraction (TF-IDF) ---")

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=3,
    max_features=10000
)

print("Fitting TF-IDF on SYNTHETIC training data...")
# Fit *only* on the synthetic training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

print("Transforming ORIGINAL evaluation data...")
# Transform the original data using the *same* vocabulary
X_eval_tfidf = tfidf_vectorizer.transform(X_eval)

print(f"TF-IDF Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# --- 6. Train the Model ---
print("\n--- Step 6: Training Model on 100% Synthetic Data ---")

model = LogisticRegression(
    random_state=RANDOM_STATE,
    solver='liblinear',
    C=1.0
    # NOTE: We do NOT use class_weight='balanced'
    # because our synthetic training set (300/300)
    # is already perfectly balanced.
)

model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# --- 7. Evaluate the Model on 100% Original Data ---
print("\n" + "=" * 30)
print("  EVALUATION ON 100% ORIGINAL DATA")
print("=" * 30)

y_pred = model.predict(X_eval_tfidf)
test_accuracy = accuracy_score(y_eval, y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"(Your original model was ~0.76)")

print("\nClassification Report (on Original Data):")
print(classification_report(y_eval, y_pred, target_names=class_names))

print("\nConfusion Matrix (on Original Data):")
print(pd.DataFrame(confusion_matrix(y_eval, y_pred),
                 index=[f'Actual: {c}' for c in class_names],
                 columns=[f'Predicted: {c}' for c in class_names]))

print("\n--- Script Complete ---")