## ---------------------------------
##  FILE: train_on_augmented_v2.py
## ---------------------------------
#
# This script trains the LogisticRegression model
# on the new, larger, augmented dataset.
#
# It also calculates and prints the
# TRAINING accuracy for comparison.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

# --- 1. Configuration ---
AUGMENTED_DATA_FILE = 'augmented_data/transcripts_augmented_mlm.csv'
RANDOM_STATE = 42

# Define split ratios
TEST_SIZE = 0.10
VAL_SIZE_OF_REMAINDER = 0.20 / (1.0 - TEST_SIZE)

print(f"--- Step 1: Loading Augmented Data ---")
try:
    df = pd.read_csv(AUGMENTED_DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{AUGMENTED_DATA_FILE}' not found.")
    print("Please run 'augment_mlm.py' first.")
    sys.exit()

print(f"Loaded {len(df)} records (original + synthetic).")

# --- 2. Preprocessing & Label Encoding ---
print("\n--- Step 2: Preprocessing ---")

df = df.dropna(subset=['Processed_Text', 'Diagnosis'])
print(f"Remaining records after dropping NaNs: {len(df)}")

X = df['Processed_Text']
y_raw = df['Diagnosis']

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

print(f"\nLabel Encoding:")
for i, class_name in enumerate(le.classes_):
    print(f"  {class_name} -> {i}")

# --- 3. Train-Validation-Test Split (70-20-10) ---
print("\n--- Step 3: Splitting Data (70-20-10) ---")

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=VAL_SIZE_OF_REMAINDER,
    random_state=RANDOM_STATE,
    stratify=y_train_val
)

print(f"Total records:   {len(X)}")
print(f"Training set:    {len(X_train)} ({len(X_train)/len(X):.1%})")
print(f"Validation set:  {len(X_val)} ({len(X_val)/len(X):.1%})")
print(f"Test set:        {len(X_test)} ({len(X_test)/len(X):.1%})")

# --- 4. Feature Extraction (TF-IDF) ---
print("\n--- Step 4: Feature Extraction (TF-IDF) ---")

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=3,
    max_features=10000
)

print("Fitting TF-IDF on training (70%) data...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

print("Transforming validation and test data...")
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# --- 5. Train the Model (Logistic Regression) ---
print("\n--- Step 5: Training Model ---")

model = LogisticRegression(
    random_state=RANDOM_STATE,
    solver='liblinear',
    C=1.0,
    class_weight='balanced'
)

print("Training Logistic Regression model on the AUGMENTED data...")
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# --- 6. Evaluate the Model ---
print("\n--- Step 6: Model Evaluation ---")
print(f"(Original model test accuracy was: ~0.76)")

# --- A. Evaluate on Training Set ---
print("---" * 10)
print("  A. TRAINING SET EVALUATION (70%)")
print("---" * 10)
y_train_pred = model.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# B. Evaluate on Validation Set
print("\n" + "---" * 10)
print("  B. VALIDATION SET EVALUATION (20%)")
print("---" * 10)
y_val_pred = model.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=class_names))

# C. Evaluate on Test Set
print("\n" + "---" * 10)
print("  C. TEST SET EVALUATION (10%)")
print("---" * 10)
y_test_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=class_names))

print("\n--- Script Complete ---")