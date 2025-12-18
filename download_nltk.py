## ---------------------------------
##  FILE: download_nltk.py (v2)
## ---------------------------------
#
# Run this script *once* to download the
# NLTK packages needed for augmentation.
#
# This version adds the specific 'averaged_perceptron_tagger_eng'
# resource to fix the LookupError.

import nltk
import ssl

print("--- Attempting to download NLTK data ---")

# This try/except block handles a common SSL error on some systems
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download the required packages
try:
    print("Downloading 'wordnet'...")
    nltk.download('wordnet')
    
    print("Downloading 'averaged_perceptron_tagger'...")
    nltk.download('averaged_perceptron_tagger')
    
    print("Downloading 'omw-1.4'...")
    nltk.download('omw-1.4')
    
    print("Downloading 'averaged_perceptron_tagger_eng' (Fixing error)...")
    nltk.download('averaged_perceptron_tagger_eng') # <-- THIS IS THE FIX
    
    print("\n--- Downloads complete! ---")
    print("You can now run 'augment_data.py'")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please check your internet connection and try again.")