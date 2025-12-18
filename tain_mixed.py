## ---------------------------------
##  FILE: train_on_mixed_data_v2.py
## ---------------------------------
#
# This script trains on a mixed 60/40 set and
# evaluates on a pure, original test set.
#
# (NEW) It now also calculates and prints
# the TRAINING accuracy.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# --- 1. Configuration ---
ORIGINAL_DATA_FILE = 'transcripts_classifier_clean.csv'
AUGMENTED_DATA_FILE = 'augmented_data/transcripts_augmented_mlm.csv'
RANDOM_STATE = 42
ORIGINAL_TEST_SET_SIZE = 0.25 # Hold out 25% of original data for final test

print(f"--- Step 1: Loading Data ---")
try:
    df_original = pd.read_csv(ORIGINAL_DATA_FILE)
    df_augmented_full = pd.read_csv(AUGMENTED_DATA_FILE)
except FileNotFoundError as e:
    print(f"Error: Could not find a file. {e}")
    sys.exit()

# Drop NaNs
df_original = df_original.dropna(subset=['Processed_Text', 'Diagnosis'])
df_augmented_full = df_augmented_full.dropna(subset=['Processed_Text', 'Diagnosis'])

print(f"Loaded {len(df_original)} original records.")
print(f"Loaded {len(df_augmented_full)} total augmented records.")

# --- 2. Separate Synthetic Data ---
print("\n--- Step 2: Separating Data ---")

# Find the 600 synthetic-only samples
df_merged = pd.merge(
    df_augmented_full,
    df_original,
    on=['Processed_Text', 'Diagnosis'],
    how='left',
    indicator=True
)
df_synthetic = df_merged[df_merged['_merge'] == 'left_only'].copy()
df_synthetic = df_synthetic[['Processed_Text', 'Diagnosis']]

print(f"Found {len(df_synthetic)} synthetic-only samples.")

# --- 3. Create Final Train and Test Sets ---
print("\n--- Step 3: Creating Mixed Train Set ---")

# 1. Split the ORIGINAL data
df_original_train, df_original_test = train_test_split(
    df_original,
    test_size=ORIGINAL_TEST_SET_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_original['Diagnosis']
)

print(f"  Held-out Original Test Set: {len(df_original_test)} samples (for final eval)")
print(f"  Original Train portion: {len(df_original_train)} samples")

# 2. Create the mixed training set
df_train_mixed = pd.concat([
    df_original_train,
    df_synthetic
], ignore_index=True)
df_train_mixed = df_train_mixed.sample(frac=1, random_state=RANDOM_STATE)

print(f"\n  New Mixed Training Set: {len(df_train_mixed)} total samples")

# 3. Define X and y
X_train = df_train_mixed['Processed_Text']
y_train_raw = df_train_mixed['Diagnosis']

X_test = df_original_test['Processed_Text']
y_test_raw = df_original_test['Diagnosis']

# --- 4. Label Encoding ---
le = LabelEncoder()
le.fit(df_original['Diagnosis'])
class_names = le.classes_

y_train = le.transform(y_train_raw)
y_test = le.transform(y_test_raw)

print(f"\nLabels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- 5. Feature Extraction (TF-IDF) ---
print("\n--- Step 5: Feature Extraction (TF-IDF) ---")

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=3,
    max_features=10000
)

print("Fitting TF-IDF on MIXED training data...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

print("Transforming ORIGINAL test data...")
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# --- 6. Train the Model ---
print("\n--- Step 6: Training Model on Mixed Data ---")

model = LogisticRegression(
    random_state=RANDOM_STATE,
    solver='liblinear',
    C=1.0,
    class_weight='balanced'
)

model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# --- 7. Evaluate the Model ---
print("\n" + "=" * 30)
print("  MODEL EVALUATION")
print("=" * 30)

# --- (NEW) A. Evaluate on Training Set ---
print("\n  A. TRAINING SET EVALUATION (Mixed Data)")
print("---" * 10)
y_train_pred = model.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# B. Evaluate on Test Set
print("\n  B. EVALUATION ON 100% ORIGINAL TEST DATA")
print("---" * 10)
y_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"(Your original, non-augmented model was ~0.76)")

print("\nClassification Report (on Original Test Data):")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\n--- Script Complete ---")