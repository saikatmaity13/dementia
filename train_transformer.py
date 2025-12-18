## ---------------------------------
##  TRAIN & EVALUATE TRANSFORMER (DistilBERT)
## ---------------------------------
#
# This script fine-tunes a pre-trained DistilBERT model.
# This is a deep learning model and will take a few minutes
# to run, even on a GPU.
#
# !! REQUIRES A GPU (like Google Colab) to run efficiently !!
#
# Required installs:
# pip install transformers datasets torch scikit-learn

import pandas as pd
import numpy as np
import sys
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Import Hugging Face specific libraries ---
try:
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
except ImportError:
    print("Error: Required libraries not found.")
    print("Please run: pip install transformers datasets torch scikit-learn")
    sys.exit()

# --- 1. Configuration ---
CLEAN_DATA_FILE = 'transcripts_classifier_clean.csv'
MODEL_NAME = "dmis-lab/biobert-v1.1"
RANDOM_STATE = 42

# Define split ratios
TEST_SIZE = 0.10
VAL_SIZE_OF_REMAINDER = 0.20 / (1.0 - TEST_SIZE)

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {device} ---")
if device == "cpu":
    print("WARNING: Running on CPU. This will be VERY slow.")
    print("It is highly recommended to run this in Google Colab with a GPU.")

# --- 2. Load and Prepare Data ---
print("\n--- Step 1: Loading Data ---")
try:
    df = pd.read_csv(CLEAN_DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{CLEAN_DATA_FILE}' not found.")
    sys.exit()

df = df.dropna(subset=['Processed_Text', 'Diagnosis'])
print(f"Loaded {len(df)} records.")

# --- 3. Label Encoding ---
# The Transformer model needs numeric labels: 0, 1
le = LabelEncoder()
df['label'] = le.fit_transform(df['Diagnosis'])
df = df.rename(columns={'Processed_Text': 'text'})

# Create a mapping for later (e.g., 0 -> 'dementia')
id2label = {i: name for i, name in enumerate(le.classes_)}
label2id = {name: i for i, name in enumerate(le.classes_)}
print(f"Labels: {label2id}")

# --- 4. Train-Validation-Test Split ---
print("\n--- Step 2: Splitting Data (70-20-10) ---")

# We split the DataFrame *before* creating the Dataset objects
df_train_val, df_test = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df['label']
)

df_train, df_val = train_test_split(
    df_train_val,
    test_size=VAL_SIZE_OF_REMAINDER,
    random_state=RANDOM_STATE,
    stratify=df_train_val['label']
)

print(f"Training set:    {len(df_train)}")
print(f"Validation set:  {len(df_val)}")
print(f"Test set:        {len(df_test)}")

# --- 5. Convert to Hugging Face Dataset ---
# This format is optimized for the Trainer
ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)
ds_test = Dataset.from_pandas(df_test)

# --- 6. Tokenization ---
print(f"\n--- Step 3: Tokenizing Data (using {MODEL_NAME}) ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # padding=True pads to the longest sequence in the batch
    # truncation=True cuts off text if it's longer than the model's max (512)
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True
    )

# Apply tokenization to all splits
tokenized_ds_train = ds_train.map(tokenize_function, batched=True)
tokenized_ds_val = ds_val.map(tokenize_function, batched=True)
tokenized_ds_test = ds_test.map(tokenize_function, batched=True)

# --- 7. Define Model and Metrics ---
print(f"\n--- Step 4: Loading Model ({MODEL_NAME}) ---")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_),
    id2label=id2label,
    label2id=label2id
)

# Helper function to compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Get the index with the highest probability
    preds = np.argmax(predictions, axis=1)
    
    # Calculate precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 8. Define Training Arguments ---
print("\n--- Step 5: Setting up Trainer ---")
training_args = TrainingArguments(
    output_dir="./transformer_results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./transformer_logs",
    logging_steps=10,
    # --- Evaluation ---
    eval_strategy="epoch", # <-- THIS IS THE FIX (was evaluation_strategy)
    save_strategy="epoch", # This one is correct
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# --- 9. Initialize Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val,
    compute_metrics=compute_metrics,
)

# --- 10. Train the Model ---
print("\n--- Step 6: Starting Model Training ---")
print("(This will take several minutes...)")
trainer.train()
print("Training complete.")

# --- 11. Evaluate on Test Set ---
print("\n" + "=" * 30)
print("  FINAL TEST SET EVALUATION")
print("=" * 30)

results = trainer.evaluate(eval_dataset=tokenized_ds_test)

print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
print(f"Test F1-Score: {results['eval_f1']:.4f}")
print(f"Test Precision: {results['eval_precision']:.4f}")
print(f"Test Recall: {results['eval_recall']:.4f}")

print("\n--- Script Complete ---")