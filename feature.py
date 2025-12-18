## ---------------------------------
##  COMPLETE SCRIPT: TRAIN, EVALUATE, & INTERPRET
## ---------------------------------
#
# This complete script does:
# 1. Loads 'transcripts_classifier_clean.csv'
# 2. Splits data (70% train, 20% val, 10% test)
# 3. Performs TF-IDF Feature Extraction
# 4. Trains a Logistic Regression model (with class_weight='balanced')
# 5. Evaluates the model on the validation and test sets
# 6. Extracts and prints the most important words (features)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

# --- 1. Configuration ---
CLEAN_DATA_FILE = 'transcripts_classifier_clean.csv'
RANDOM_STATE = 42 # For reproducible results

# Define split ratios
TEST_SIZE = 0.10  # 10% for test set
VAL_SIZE_OF_REMAINDER = 0.20 / (1.0 - TEST_SIZE) # 20% for validation (from original)

print("--- Step 1: Loading Data ---")
try:
    df = pd.read_csv(CLEAN_DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{CLEAN_DATA_FILE}' not found.")
    print("Please run 'clean_data.py' first, or make sure the file is in the same folder.")
    sys.exit()

print(f"Loaded {len(df)} records.")

# --- 2. Preprocessing & Label Encoding ---
print("\n--- Step 2: Preprocessing ---")

# Drop rows with missing text
df = df.dropna(subset=['Processed_Text', 'Diagnosis'])
print(f"Remaining records after dropping NaNs: {len(df)}")

# Create features (X) and target (y)
X = df['Processed_Text']
y_raw = df['Diagnosis']

# Encode string labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_ # Save class names for reports

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

print("Fitting TF-IDF on training data...")
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
    class_weight='balanced'  # <-- This was the important fix
)

print("Training Logistic Regression model (with class_weight='balanced')...")
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# --- 6. Evaluate the Model ---
print("\n--- Step 6: Model Evaluation ---")

# A. Evaluate on Validation Set
print("---" * 10)
print("  VALIDATION SET EVALUATION")
print("---" * 10)
y_val_pred = model.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=class_names))

# B. Evaluate on Test Set
print("\n" + "---" * 10)
print("  TEST SET EVALUATION")
print("---" * 10)
y_test_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=class_names))

# --- 7. Feature Importance ---
print("\n" + "=" * 30)
print("  MODEL INTERPRETATION")
print("=" * 30)

# Check if the model is LogisticRegression and has coefficients
if hasattr(model, 'coef_'):
    # Get the feature names from the vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Get the coefficients (importance scores) from the model
    # model.coef_[0] contains the scores for the "positive" class
    coefs = model.coef_[0]
    
    # Create a dataframe of words and their scores
    word_scores = pd.DataFrame({
        'word': feature_names,
        'score': coefs
    })
    
    # --- Sort by score to find top words ---
    
    # The "positive" class (score > 0)
    # This will be the class encoded as '1' (e.g., 'nodementia')
    top_positive_label = le.classes_[1]
    top_positive_words = word_scores.sort_values('score', ascending=False).head(20)

    # The "negative" class (score < 0)
    # This will be the class encoded as '0' (e.g., 'dementia')
    top_negative_label = le.classes_[0]
    # Multiply scores by -1 so highest numbers (most negative) are on top
    word_scores['score'] = word_scores['score'] * -1
    top_negative_words = word_scores.sort_values('score', ascending=False).head(20)

    print(f"\nTop 20 words associated with '{top_positive_label}':")
    print(top_positive_words.to_string(index=False))
    
    print(f"\nTop 20 words associated with '{top_negative_label}':")
    print(top_negative_words[['word', 'score']].to_string(index=False))

else:
    print("\nCould not extract feature importance.")
    print("This code works for LogisticRegression and LinearSVC.")

print("\n--- Script Complete ---")