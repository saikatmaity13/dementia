## ---------------------------------
##  FILE: extract_paralinguistic_features.py
## ---------------------------------
#
# This script extracts PARALINGUISTIC features:
# 1. Pitch (F0) - To detect monotone voice
# 2. Speech Rate - Words per second (requires transcripts.csv)
# 3. Pause Frequency - How often they stop
#
# It merges data from 'transcripts.csv' and the .wav files.

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_ROOT_FOLDER = Path("C:/dementia_data")
TRANSCRIPT_FILE = "transcripts.csv"
OUTPUT_FILE = "paralinguistic_features.csv"

# --- HELPER FUNCTION ---
def extract_pitch_and_pauses(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. Duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. Pitch (Fundamental Frequency - F0) using PyIN
        # This is heavy, so we use a frame_length usually suited for speech
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7')
        )
        
        # Filter out NaNs (unvoiced parts) to get pitch of actual speech
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) > 0:
            pitch_mean = np.mean(valid_f0)
            pitch_std = np.std(valid_f0) # Low std = Monotone voice
            pitch_range = np.max(valid_f0) - np.min(valid_f0)
        else:
            pitch_mean = 0
            pitch_std = 0
            pitch_range = 0
            
        # 3. Pause Frequency (Count of pauses)
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        # The number of pauses is roughly the number of non-silent chunks - 1
        n_pauses = max(0, len(non_silent_intervals) - 1)
        
        # Pause rate = Pauses per second
        pause_rate = n_pauses / duration if duration > 0 else 0
        
        return {
            'duration': duration,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,   # Critical feature for dementia
            'pitch_range': pitch_range,
            'pause_count': n_pauses,
            'pause_rate': pause_rate
        }

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

# --- MAIN SCRIPT ---

print("--- Step 1: Loading Transcripts for Speech Rate ---")
try:
    df_text = pd.read_csv(TRANSCRIPT_FILE)
    # Create a helper dictionary: filename -> word_count
    # We calculate word count from the transcript text
    df_text['word_count'] = df_text['Transcript'].fillna("").apply(lambda x: len(str(x).split()))
    
    # Map File_Name to Word Count for easy lookup
    # Ensure File_Name matches exactly (sometimes extensions differ)
    word_count_map = dict(zip(df_text['File_Name'], df_text['word_count']))
    diagnosis_map = dict(zip(df_text['File_Name'], df_text['Diagnosis']))
    
    print(f"Loaded transcripts for {len(df_text)} files.")
    
except FileNotFoundError:
    print(f"Error: '{TRANSCRIPT_FILE}' not found.")
    print("Cannot calculate Speech Rate without transcripts.")
    print("Please run 'transcribe_audio.py' first.")
    sys.exit()

print("\n--- Step 2: Extracting Audio Features (Pitch & Pauses) ---")
all_data = []
all_wav_files = list(DATA_ROOT_FOLDER.rglob("*.wav"))
total_files = len(all_wav_files)

for i, audio_path in enumerate(all_wav_files):
    file_name = audio_path.name
    
    if (i + 1) % 10 == 0:
        print(f"[{i+1}/{total_files}] Processing...")
        
    # 1. Get Audio Features
    audio_feats = extract_pitch_and_pauses(audio_path)
    
    if audio_feats:
        # 2. Get Text Features (Word Count)
        # We try to find the file in our transcript map
        word_cnt = word_count_map.get(file_name)
        
        # If not found directly, try matching without extension? 
        # (Optional logic, but usually filenames match if pipeline is consistent)
        if word_cnt is None:
            word_cnt = 0 # Fallback
            
        # 3. Calculate Speech Rate (Words per Second)
        duration = audio_feats['duration']
        speech_rate = word_cnt / duration if duration > 0 else 0
        
        # Combine everything
        row = {
            'File_Name': file_name,
            'Diagnosis': diagnosis_map.get(file_name, 'unknown'),
            'word_count': word_cnt,
            'speech_rate': speech_rate, # Critical feature
            **audio_feats, # Unpack audio features
            'Full_Path': str(audio_path)
        }
        all_data.append(row)

# --- SAVE ---
df_final = pd.DataFrame(all_data)

# Reorder for cleanliness
cols = ['Diagnosis', 'File_Name', 'speech_rate', 'pitch_std', 'pause_rate', 'word_count', 'duration']
other = [c for c in df_final.columns if c not in cols and c != 'Full_Path']
df_final = df_final[cols + other + ['Full_Path']]

df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\nSuccessfully saved paralinguistic features to '{OUTPUT_FILE}'")
print(f"Total processed: {len(df_final)}")