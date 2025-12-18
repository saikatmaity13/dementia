## ---------------------------------
##  FILE: augment_mlm.py
## ---------------------------------
#
# This script generates 300 new, MEANINGFUL samples per class
# using a Masked Language Model (MLM) like DistilRoBERTa.
#
# This is much higher quality than random swaps/deletions.
#
# This script requires:
# pip install transformers torch pandas

import pandas as pd
import os
import random
import torch
import sys

# --- Import Transformers Libraries ---
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM
    )
except ImportError:
    print("Error: 'transformers' library not found.")
    print("Please run: pip install transformers torch")
    sys.exit()

# --- Configuration ---
INPUT_FILE = 'transcripts_classifier_clean.csv'
OUTPUT_FOLDER = 'augmented_data'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'transcripts_augmented_mlm.csv')

# List of all classes you want to augment
CLASSES_TO_AUGMENT = ['dementia', 'nodementia']
# Target number of *new* samples to generate for each class
TARGET_SAMPLES_PER_CLASS = 300 

# Model to use for predictions
MODEL_NAME = 'distilroberta-base' 

# How many words to mask (as a fraction)
MASK_PROBABILITY = 0.15 
# How many top predictions to choose from
TOP_K = 5 

# --- Setup Device (GPU if available) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {device} ---")
if device == "cpu":
    print("WARNING: Running on CPU. This will be slow.")

# --- 1. Load Model and Tokenizer ---
print(f"Loading model '{MODEL_NAME}'... (This may take a moment)")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    model.eval() # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection or the model name.")
    sys.exit()

# --- 2. Augmentation Function ---
def augment_text_mlm(text, tokenizer, model, device, prob=0.15, top_k=5):
    """Generates a new sentence by masking and predicting words."""
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'][0]
        
        # Don't mask special tokens [CLS] and [SEP]
        non_special_tokens_mask = (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
        
        # Determine which tokens to mask
        mask = torch.rand(input_ids.shape) < prob
        
        # Combine the masks
        mask &= non_special_tokens_mask
        
        # If no tokens were selected, randomly pick one (but not special tokens)
        if not mask.any():
            non_special_indices = torch.where(non_special_tokens_mask)[0]
            if len(non_special_indices) > 0:
                rand_idx = random.choice(non_special_indices.tolist())
                mask[rand_idx] = True
            else:
                # Text is too short or all special tokens
                return None 

        # Create a copy of input_ids to modify
        masked_input_ids = input_ids.clone()
        # Apply the mask
        masked_input_ids[mask] = tokenizer.mask_token_id
        
        # Move to device and add batch dimension
        masked_input_ids = masked_input_ids.unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(masked_input_ids)
            logits = outputs.logits
            
        # Get predictions for the masked tokens
        masked_indices = torch.where(mask)[0]
        
        # Loop through each masked token and replace it
        new_input_ids = input_ids.clone()
        
        for idx in masked_indices:
            # Get the top k predictions for this specific token
            top_k_logits, top_k_indices = torch.topk(logits[0, idx, :], top_k)
            
            # Randomly choose one of the top k predictions
            chosen_token_id = random.choice(top_k_indices.tolist())
            
            # Replace the original token with the new predicted token
            new_input_ids[idx] = chosen_token_id
            
        # Decode the new IDs back into text
        new_text = tokenizer.decode(new_input_ids, skip_special_tokens=True)
        
        # Don't return identical or empty text
        if new_text != text and new_text.strip():
            return new_text
        else:
            return None

    except Exception as e:
        print(f"Error in augmentation: {e}")
        return None

# --- 3. Main Script Logic ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"--- Output will be saved to '{OUTPUT_FOLDER}' folder ---")

print(f"--- Loading data from '{INPUT_FILE}' ---")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: '{INPUT_FILE}' not found.")
    sys.exit()

df = df.dropna(subset=['Processed_Text', 'Diagnosis'])
print(f"Loaded {len(df)} total original records.")

# Keep a copy of the original data
original_df = df.copy()

# This list will hold all our new, augmented data
all_new_data = []

# Loop over all classes you want to augment
for class_label in CLASSES_TO_AUGMENT:
    
    print(f"\n--- Augmenting class: '{class_label}' ---")
    # Get all original texts for this class
    original_texts = df[df['Diagnosis'] == class_label]['Processed_Text'].tolist()
    
    if not original_texts:
        print(f"No text found for class '{class_label}'. Skipping.")
        continue

    print(f"Found {len(original_texts)} original samples.")
    print(f"Generating {TARGET_SAMPLES_PER_CLASS} new meaningful samples...")

    num_generated = 0
    new_samples_for_class = []

    # Keep looping until we hit our target
    while num_generated < TARGET_SAMPLES_PER_CLASS:
        # Pick a random original text to modify
        random_text = random.choice(original_texts)
        
        # Generate a new text
        new_text = augment_text_mlm(
            random_text, tokenizer, model, device,
            prob=MASK_PROBABILITY, top_k=TOP_K
        )
        
        if new_text:
            new_samples_for_class.append({
                'Diagnosis': class_label,
                'Processed_Text': new_text
            })
            num_generated += 1
            
            # Print progress
            if num_generated % 25 == 0:
                print(f"  Generated {num_generated}/{TARGET_SAMPLES_PER_CLASS} for {class_label}...")

    all_new_data.extend(new_samples_for_class)
    print(f"--- Finished generating {num_generated} samples for {class_label} ---")

# --- 4. Combine and Save ---
df_augmented = pd.DataFrame(all_new_data)
print(f"\n--- Total new synthetic samples created: {len(df_augmented)} ---")

# Combine the ORIGINAL data with the NEW augmented data
df_final = pd.concat([original_df, df_augmented], ignore_index=True)

# Shuffle the final dataset
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nOriginal dataset size: {len(original_df)}")
print(f"New augmented dataset size: {len(df_final)}")

# Save the new dataset to the specified folder
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\n--- Successfully saved meaningful augmented data to '{OUTPUT_FILE}' ---")