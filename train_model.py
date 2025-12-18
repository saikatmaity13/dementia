## ---------------------------------
##  TRAIN & EVALUATE ADVANCED MODELS
## ---------------------------------
#
# This script will test two new models on the same data:
# 1. LinearSVC (A top-tier linear model for text)
# 2. RandomForestClassifier (A tree-based model)
#
# It uses the same 70/20/10 split to make the results
# comparable to the Logistic Regression model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# --- 1. Configuration ---
CLEAN_DATA_FILE = 'transcripts_classifier_clean.csv'
RANDOM_STATE = 42

TEST_SIZE = 0.10
VAL_SIZE_OF_REMAINDER = 0.20 / (1.0 - TEST_SIZE)

print("--- Step 1: Loading Data ---")
try:
    df = pd.read_csv(CLEAN_DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{CLEAN_DATA_FILE}' not found.")
    sys.exit()

df = df.dropna(subset=['Processed_Text', 'Diagnosis'])
print(f"Loaded {len(df)} records.")

# --- 2. Preprocessing, Encoding & Split ---
print("\n--- Step 2: Preparing Data ---")
X = df['Processed_Text']
y_raw = df['Diagnosis']

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

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

print(f"Training set:    {len(X_train)}")
print(f"Validation set:  {len(X_val)}")
print(f"Test set:        {len(X_test)}")

# --- 3. Feature Extraction (TF-IDF) ---
print("\n--- Step 3: Feature Extraction (TF-IDF) ---")
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=3,
    max_features=10000
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")


# --- 4. Train and Evaluate Models ---
print("\n--- Step 4: Training & Evaluating Models ---")

# Create a list of models to try
models_to_try = [
    (
        "LinearSVC",
        LinearSVC(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            C=1.0,  # Regularization parameter
            dual=False # Recommended when n_samples > n_features
        )
    ),
    (
        "RandomForest",
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_estimators=100  # Number of trees
        )
    )
]

# Dictionary to store results
results = {}

for name, model in models_to_try:
    print("\n" + "=" * 30)
    print(f"  TRAINING: {name}")
    print("=" * 30)
    
    # Train the model
    model.fit(X_train_tfidf, y_train)
    print("Training complete.")
    
    # Evaluate on Validation Set
    print("\n  Validation Results:")
    y_val_pred = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(classification_report(y_val, y_val_pred, target_names=class_names, zero_division=0))
    
    # Evaluate on Test Set
    print("  Test Results:")
    y_test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))
    
    # Store results
    results[name] = {
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }

# --- 5. Final Comparison ---
print("\n" + "=" * 30)
print("  FINAL MODEL COMPARISON")
print("=" * 30)
print("\n(Your previous Logistic Regression Test Accuracy was: 0.7600)")

for name, res in results.items():
    print(f"\nModel: {name}")
    print(f"  Validation Accuracy: {res['val_accuracy']:.4f}")
    print(f"  Test Accuracy:       {res['test_accuracy']:.4f}")

print("\n--- Script Complete ---")