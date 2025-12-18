## ---------------------------------
##  FILE: clean_data_v2.py
## ---------------------------------
#
# This script re-cleans the data but KEEPS the 'File_Name'
# column so we can merge it with the audio features later.

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import sys

# --- CONFIGURATION ---
INPUT_CSV_NAME = 'transcripts.csv'
OUTPUT_CSV_NAME = 'transcripts_with_ids.csv' # New filename

# --- NLTK SETUP ---
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    print("Error: NLTK data missing. Run 'download_nltk.py' first.")
    sys.exit()

def clean_text_for_classifier(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    processed_tokens = []
    for word in tokens:
        if word not in stop_words and len(word) > 1:
            processed_tokens.append(lemmatizer.lemmatize(word))
    return ' '.join(processed_tokens)

# --- MAIN SCRIPT ---
print("Loading transcripts...")
try:
    df = pd.read_csv(INPUT_CSV_NAME)
except FileNotFoundError:
    print(f"Error: '{INPUT_CSV_NAME}' not found. Please run 'transcribe_audio.py' first.")
    sys.exit()

print("Cleaning text...")
# Apply cleaning
df['Processed_Text'] = df['Transcript'].fillna('').apply(clean_text_for_classifier)

# Keep only the columns we need for the final model
df_final = df[['Diagnosis', 'Processed_Text', 'File_Name']]

df_final.to_csv(OUTPUT_CSV_NAME, index=False)
print(f"Success! Saved cleaned data with IDs to '{OUTPUT_CSV_NAME}'")