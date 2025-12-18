import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import sys
import os

# --- 1. CONFIGURATION ---
INPUT_CSV_NAME = 'transcripts.csv'
OUTPUT_CSV_NAME = 'transcripts_classifier_clean.csv' # Renamed output
# TRAINING_TEXT_FILE = 'training_data.txt' # This file is no longer needed here

# --- Download NLTK data (uncomment if first time) ---
# print("Downloading NLTK data...")
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# print("Downloads complete.")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# This is your "aggressive" cleaning function for the classifier
# We are keeping it exactly as-is because it works well for TF-IDF.
def clean_text_for_classifier(text):
    """Applies an 'aggressive' cleaning process for classification."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove all non-alphabetic characters (brackets, numbers, etc.)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    processed_tokens = []
    for word in tokens:
        # Reduced junk words, as the regex handles most of it.
        if word not in stop_words and len(word) > 1:
            processed_tokens.append(lemmatizer.lemmatize(word))
    return ' '.join(processed_tokens)

# --- 2. MAIN SCRIPT ---
print("--- Step 1: Loading and Cleaning for CLASSIFIER ---")

# --- 3. LOAD YOUR DATA ---
print(f"Loading '{INPUT_CSV_NAME}'...")
try:
    df = pd.read_csv(INPUT_CSV_NAME)
except FileNotFoundError:
    print(f"Error: '{INPUT_CSV_NAME}' not found.")
    print("Please run 'transcribe_audio.py' first.")
    sys.exit()
print(f"Loaded {len(df)} transcripts.")

# --- 4. APPLY THE CLEANING ---
print("Applying aggressive cleaning for classifier...")
df['Processed_Text'] = df['Transcript'].fillna('').apply(clean_text_for_classifier)
print("Classifier cleaning complete.")

# --- 5. SAVE THE PROCESSED FILE ---
# Keep only the columns we need
df_final = df[['Diagnosis', 'Processed_Text']]
df_final.to_csv(OUTPUT_CSV_NAME, index=False)

print(f"Successfully saved classifier-ready data to '{OUTPUT_CSV_NAME}'!")