## ---------------------------------
##  FILE: extract_acoustic_features.py
## ---------------------------------
#
# This script extracts acoustic and paralinguistic features
# from your .wav files using the 'librosa' library.
#
# It saves a new CSV file: 'acoustic_features.csv'

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. SETTINGS ---
# Update this to match your folder path (same as in transcribe_audio.py)
DATA_ROOT_FOLDER = Path("C:/dementia_data") 
OUTPUT_CSV_NAME = "acoustic_features.csv"

# --- 2. HELPER FUNCTIONS ---

def extract_features(file_path):
    """
    Extracts specific acoustic features from a single audio file.
    """
    try:
        # Load audio file (sr=None preserves original sampling rate)
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. Duration (in seconds)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. Pause / Silence Features
        # Detect non-silent chunks
        non_silent_intervals = librosa.effects.split(y, top_db=20) # top_db controls sensitivity
        
        # Calculate total silence duration
        non_silent_duration = 0
        for start, end in non_silent_intervals:
            non_silent_duration += (end - start) / sr
            
        silence_duration = duration - non_silent_duration
        silence_fraction = silence_duration / duration if duration > 0 else 0
        
        # 3. Root Mean Square Energy (Loudness)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # 4. Zero Crossing Rate (Noisiness/Roughness)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 5. Spectral Centroid (Brightness of voice)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        
        # 6. MFCCs (Timbre / Voice Quality)
        # We extract 13 coefficients which is standard for speech
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1) # Mean of each coefficient
        
        # --- Pack features into a dictionary ---
        features = {
            'duration': duration,
            'silence_duration': silence_duration,
            'silence_fraction': silence_fraction,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'spectral_centroid': cent_mean,
        }
        
        # Add each MFCC coefficient as a separate column
        for i, val in enumerate(mfccs_mean):
            features[f'mfcc_{i+1}'] = val
            
        return features

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

# --- 3. MAIN SCRIPT ---

if not DATA_ROOT_FOLDER.exists():
    print(f"Error: Path not found: {DATA_ROOT_FOLDER}")
    sys.exit()

all_data = []
print(f"Scanning '{DATA_ROOT_FOLDER}' for .wav files...")
all_wav_files = list(DATA_ROOT_FOLDER.rglob("*.wav"))
total_files = len(all_wav_files)

print(f"Found {total_files} files. Starting extraction (this may take some time)...")

for i, audio_file_path in enumerate(all_wav_files):
    file_name = audio_file_path.name
    # Assumes folder structure: root -> dementia/nodementia -> file.wav
    diagnosis_label = audio_file_path.parent.parent.name 
    
    # Print progress every 10 files
    if (i + 1) % 10 == 0:
        print(f"[{i+1}/{total_files}] Processing...")

    # Extract features
    features = extract_features(audio_file_path)
    
    if features:
        # Add ID info
        features['File_Name'] = file_name
        features['Diagnosis'] = diagnosis_label
        features['Full_Path'] = str(audio_file_path)
        
        all_data.append(features)

# --- 4. SAVE RESULTS ---
if len(all_data) > 0:
    df_features = pd.DataFrame(all_data)
    
    # Reorder columns to put IDs first
    cols = ['Diagnosis', 'File_Name', 'duration', 'silence_fraction'] 
    # Add the rest of the columns dynamically
    other_cols = [c for c in df_features.columns if c not in cols and c != 'Full_Path']
    df_features = df_features[cols + other_cols + ['Full_Path']]
    
    df_features.to_csv(OUTPUT_CSV_NAME, index=False)
    print(f"\n🎉 Success! Features saved to '{OUTPUT_CSV_NAME}'.")
    print(f"Extracted features for {len(df_features)} files.")
    print("\nNext Step: You can now train a model on this CSV, just like you did with the text!")
else:
    print("No data extracted. Check your folder paths.")