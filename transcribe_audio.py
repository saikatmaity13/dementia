import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
from pathlib import Path
import whisper
import re
import sys
import time

# --- 1. SETTINGS: PLEASE UPDATE THESE ---

# Set this to the folder that CONTAINS your 'dementia' 
# and 'nondementia' folders.
DATA_ROOT_FOLDER = Path("C:/dementia_data") 

# --- (MODIFICATION) ---
# "base" is fast but "medium" is much more accurate, which is
# critical for capturing the nuances of speech.
# "tiny", "base", "small", "medium", "large"
MODEL_NAME = "medium" 
# --- (END MODIFICATION) ---

OUTPUT_CSV_NAME = "transcripts.csv"
# ---------------------------------------------

# Regex to get Speaker ID (e.g., "AbeBurrows") from "AbeBurrows_5.wav"
# It assumes the ID is everything before the last underscore.
speaker_pattern = re.compile(r'^(.*)_[\d]+\.wav$')
task_id_pattern = re.compile(r'_(\d+)\.wav$')

# --- 2. CHECK FOLDER AND LOAD MODEL ---

if not DATA_ROOT_FOLDER.exists() or not DATA_ROOT_FOLDER.is_dir():
    print(f"Error: Path not found: {DATA_ROOT_FOLDER}")
    print("Please update the 'DATA_ROOT_FOLDER' variable.")
    sys.exit()

print(f"Loading Whisper model '{MODEL_NAME}' (This may take a moment)...")
model = whisper.load_model(MODEL_NAME)
print("Model loaded successfully.")

# --- 3. SCAN, TRANSCRIBE, AND COLLECT DATA ---

all_file_data = []
start_time = time.time()

# Find all .wav files in all subfolders
print("Finding all .wav files...")
all_wav_files = list(DATA_ROOT_FOLDER.rglob("*.wav"))
total_files = len(all_wav_files)

if total_files == 0:
    print(f"Error: No .wav files were found in {DATA_ROOT_FOLDER}.")
    sys.exit()
    
print(f"Found {total_files} audio files. Starting transcription...")

for i, audio_file_path in enumerate(all_wav_files):
    
    file_name = audio_file_path.name
    
    # --- A: Get the Label (Diagnosis) from the folder ---
    # (e.g., 'dementia' or 'nondementia')
    diagnosis_label = audio_file_path.parent.parent.name
    
    # --- B: Get Speaker_ID and Task_ID from filename ---
    speaker_id = ''
    task_id = ''
    
    speaker_match = speaker_pattern.search(file_name)
    if speaker_match:
        speaker_id = speaker_match.group(1) # e.g., "AbeBurrows"
        
    task_match = task_id_pattern.search(file_name.lower())
    if task_match:
        task_id = task_match.group(1) # e.g., '5'
        
    print(f"\n[{i + 1}/{total_files}] Processing: {diagnosis_label} / {file_name}")

    try:
        # --- C: Transcribe the file ---
        print("  Transcribing...")
        # fp16=False is good for CPU. If you have a good GPU, you can set this to True.
        result = model.transcribe(str(audio_file_path), fp16=False)
        transcript_text = result['text'].strip()
        print(f"  Done.")

        # --- D: Add all data to our list ---
        file_data_row = {
            'Diagnosis': diagnosis_label,
            'Speaker_ID': speaker_id,
            'Task_ID': task_id,
            'Transcript': transcript_text,
            'File_Name': file_name,
            'Full_Path': str(audio_file_path.resolve()),
        }
        all_file_data.append(file_data_row)

    except Exception as e:
        print(f"  !! FAILED to process {file_name}: {e}")
        all_file_data.append({
            'Diagnosis': diagnosis_label, 'Speaker_ID': speaker_id, 'Task_ID': task_id,
            'Transcript': f"ERROR: {e}", 'File_Name': file_name,
            'Full_Path': str(audio_file_path.resolve()),
        })

# --- 4. CREATE AND SAVE THE FINAL CSV FILE ---
print("\n---\nAll files processed.")
df = pd.DataFrame(all_file_data)

# Reorder columns
df = df[[
    'Diagnosis', 
    'Speaker_ID', 
    'Task_ID', 
    'Transcript', 
    'File_Name', 
    'Full_Path'
]]

df.to_csv(OUTPUT_CSV_NAME, index=False)
print(f"🎉 Successfully created '{OUTPUT_CSV_NAME}'!")
print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes.")