#!/usr/bin/env python
# coding: utf-8

# IMPORTING LIBRARIES


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# IMPORT DATA


kaggle_path = "/kaggle/input/dementia-kaggle/dementia_patients_health_data.csv"


dementia_kaggle = pd.read_csv("/kaggle/input/data-dementia-numerical/dementia_patients_health_data.csv")


print("\n Dementia Kaggle Dataset Overview:")
print(dementia_kaggle.info())
print("\nPreview:")
print(dementia_kaggle.head())


# Identify categorical columns
categorical_cols = dementia_kaggle.select_dtypes(include=['object']).columns

# Count unique values for each categorical column
print(" Unique category counts per categorical feature:\n")
for col in categorical_cols:
    unique_count = dementia_kaggle[col].nunique()
    print(f"{col:30}: {unique_count} unique values")

# Optional: Display unique values (for quick inspection)
print("\n Sample unique values per column (first 5 shown):\n")
for col in categorical_cols:
    print(f"{col}: {dementia_kaggle[col].unique()[:5]}")


for col in dementia_kaggle.columns:
    if dementia_kaggle[col].dtype == 'object':
        dementia_kaggle[col].fillna(dementia_kaggle[col].mode()[0], inplace=True)
    else:
        dementia_kaggle[col].fillna(dementia_kaggle[col].median(), inplace=True)


dementia_kaggle.drop_duplicates(inplace=True)


for col in dementia_kaggle.select_dtypes(include=[np.number]).columns:
    Q1 = dementia_kaggle[col].quantile(0.25)
    Q3 = dementia_kaggle[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    dementia_kaggle[col] = np.clip(dementia_kaggle[col], lower, upper)


print("\n Final Clean Dataset Summary:")
print(f"Rows: {dementia_kaggle.shape[0]}, Columns: {dementia_kaggle.shape[1]}")
print("\nSample Data After Cleaning:")
print(dementia_kaggle.head())


label_encode_cols = ['Education_Level', 'Physical_Activity']  # ordinal
le = LabelEncoder()
for col in label_encode_cols:
    dementia_kaggle[col] = le.fit_transform(dementia_kaggle[col])

categorical_cols = dementia_kaggle.select_dtypes(include=['object']).columns
dementia_kaggle = pd.get_dummies(dementia_kaggle, columns=categorical_cols, drop_first=True)


print("\n Encoding completed successfully!")
print(f"Final dataset shape: {dementia_kaggle.shape}\n")

print(" Updated column list:")
print(dementia_kaggle.columns.tolist())

pd.set_option('display.max_columns', None)
print(dementia_kaggle.head())


print(dementia_kaggle['Dementia'].value_counts())


X = dementia_kaggle.drop('Dementia', axis=1)
y = dementia_kaggle['Dementia']

# First split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("\n Dataset split summary:")
print(f"Training set:   {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set:       {X_test.shape}")


scaler = MinMaxScaler()

# Fit the scaler on training data only
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test sets using the same scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#  Convert scaled arrays back to DataFrames
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

#  Reattach target columns
train_final = pd.concat([X_train_scaled_df, y_train.reset_index(drop=True)], axis=1)
val_final = pd.concat([X_val_scaled_df, y_val.reset_index(drop=True)], axis=1)
test_final = pd.concat([X_test_scaled_df, y_test.reset_index(drop=True)], axis=1)

# Summary
print(" Normalization (MinMaxScaler) complete!")
print(f"Training set shape: {train_final.shape}")
print(f"Validation set shape: {val_final.shape}")
print(f"Test set shape: {test_final.shape}")
print("\n Sample normalized training data:")
print(train_final.head())


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#  Ensure 'Dementia' is in dataset
target_col = 'Dementia'
assert target_col in dementia_kaggle.columns, "Target column 'Dementia' not found!"

#  Compute correlation matrix
corr_matrix = dementia_kaggle.corr()

#  Get top correlations with target
target_corr = corr_matrix[target_col].sort_values(ascending=False)

print("\n Feature Correlation with Target (Dementia):")
print(target_corr)

#  Optional: visualize correlation heatmap (top 15 features)
top_features = target_corr.abs().sort_values(ascending=False).head(15).index
plt.figure(figsize=(10,8))
sns.heatmap(dementia_kaggle[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title(" Top Correlated Features with Dementia")
plt.show()

#  Barplot for clarity
plt.figure(figsize=(8,5))
target_corr.drop(target_col).sort_values(ascending=True).plot(kind='barh', color='teal')
plt.title("Feature Correlation with Target (Dementia)")
plt.xlabel("Correlation Coefficient")
plt.show()


# Manual
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

X = dementia_kaggle.drop(columns=[target_col])
y = dementia_kaggle[target_col]

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\n Mutual Information Scores with Dementia:")
print(mi_series.head(15))

# Plot
plt.figure(figsize=(8,5))
mi_series.head(15).sort_values().plot(kind='barh', color='royalblue')
plt.title("Top 15 Features by Mutual Information with Dementia")
plt.xlabel("Mutual Information Score")
plt.show()


# Tree
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n Random Forest Feature Importances (Top 15):")
print(importances.head(15))

# Plot
importances.head(15).sort_values().plot(kind='barh', color='forestgreen', figsize=(8,5))
plt.title("Top 15 Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.show()


# FEATURE SELECTION ON NORMALIZED DATA (Fixed Import Error)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel   # ✅ Missing import added

#  Fit Random Forest on normalized training data
rf_selector = RandomForestClassifier(random_state=42)
rf_selector.fit(X_train_scaled_df, y_train)

#  Compute feature importances
importances = pd.Series(rf_selector.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n🌟 Random Forest Feature Importances (on normalized data):")
print(importances)

#  Select features > 0.02 threshold
selector = SelectFromModel(rf_selector, threshold=0.02, prefit=True)
X_train_selected = selector.transform(X_train_scaled_df)
X_val_selected = selector.transform(X_val_scaled_df)
X_test_selected = selector.transform(X_test_scaled_df)

#  Get selected feature names
selected_features = X.columns[selector.get_support()]
print(f"\n✅ Selected Important Features (importance > 0.02): {selected_features.tolist()}")
print(f"Number of features selected: {len(selected_features)} / {X.shape[1]}")

#  Optional: visualize importances
plt.figure(figsize=(8,5))
importances.plot(kind='barh', color=['teal' if imp >= 0.02 else 'lightgray' for imp in importances])
plt.axvline(0.02, color='red', linestyle='--', label='Threshold = 0.02')
plt.title('Feature Importances (Normalized Data)')
plt.xlabel('Importance Score')
plt.legend()
plt.tight_layout()
plt.show()


#  NAIVE BAYES CLASSIFIER FOR DEMENTIA DETECTION

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize Naive Bayes model
nb_model = GaussianNB()

#  Train the model
nb_model.fit(X_train_selected, y_train)

#  Training Performance
y_train_pred = nb_model.predict(X_train_selected)
train_acc = accuracy_score(y_train, y_train_pred)

print("\n TRAINING SET PERFORMANCE (Naive Bayes):")
print(f"Accuracy: {train_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

#  Validation Performance
y_val_pred = nb_model.predict(X_val_selected)
val_acc = accuracy_score(y_val, y_val_pred)

print("\n VALIDATION SET PERFORMANCE (Naive Bayes):")
print(f"Accuracy: {val_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

#  Test Performance
y_test_pred = nb_model.predict(X_test_selected)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n TEST SET PERFORMANCE (Naive Bayes):")
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

#  Summary Table
import pandas as pd

summary_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Accuracy": [train_acc, val_acc, test_acc]
})

print("\n Overall Accuracy Summary:")
print(summary_df.to_string(index=False))


#  LOGISTIC REGRESSION CLASSIFIER FOR DEMENTIA DETECTION

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=500, solver='lbfgs')

#  Train the model
log_reg.fit(X_train_selected, y_train)

# TRAINING PERFORMANCE
y_train_pred = log_reg.predict(X_train_selected)
train_acc = accuracy_score(y_train, y_train_pred)

print("\n TRAINING SET PERFORMANCE (Logistic Regression):")
print(f"Accuracy: {train_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

# VALIDATION PERFORMANCE
y_val_pred = log_reg.predict(X_val_selected)
val_acc = accuracy_score(y_val, y_val_pred)

print("\n VALIDATION SET PERFORMANCE (Logistic Regression):")
print(f"Accuracy: {val_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# TEST PERFORMANCE
y_test_pred = log_reg.predict(X_test_selected)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n TEST SET PERFORMANCE (Logistic Regression):")
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# SUMMARY TABLE
summary_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Accuracy": [train_acc, val_acc, test_acc]
})

print("\n OVERALL ACCURACY SUMMARY (Logistic Regression):")
print(summary_df.to_string(index=False))


# Generate 3,000 synthetic samples (class-conditional GMM)
# Train only on synthetic; validate on original real data

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#  safety checks (must exist from your previous steps)
for var_name in ["selected_features", "X_train_selected", "X_val_selected", "y_train", "y_val"]:
    assert var_name in globals(), f"{var_name} not found. Run your preprocessing/selection cells first."

# helper: sample synthetic data with class-conditional GMM 
def generate_gmm_synthetic(X_train_sel, y_train, n_samples_total=3000, n_components=3, random_state=42):
    """
    Fit a GMM per class on normalized & selected features, then sample a total of n_samples_total.
    Class ratios follow the real training distribution.
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y_train, return_counts=True)
    class_probs = counts / counts.sum()

    # how many samples per class (round, then fix remainder)
    n_per_class = np.floor(class_probs * n_samples_total).astype(int)
    remainder = n_samples_total - n_per_class.sum()
    # assign remainder to classes with largest residuals
    residuals = (class_probs * n_samples_total) - n_per_class
    for idx in np.argsort(residuals)[::-1][:remainder]:
        n_per_class[idx] += 1

    X_synth_list, y_synth_list = [], []
    for c, n_c in zip(classes, n_per_class):
        X_c = X_train_sel[y_train.values == c]
        # Fit a small GMM; regularize covariance to avoid degenerate fits
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state
        )
        gmm.fit(X_c)
        X_c_synth, _ = gmm.sample(n_c)
        # since original features are MinMax-scaled, keep within [0,1]
        X_c_synth = np.clip(X_c_synth, 0.0, 1.0)

        X_synth_list.append(X_c_synth)
        y_synth_list.append(np.full(n_c, c, dtype=int))

    X_synth = np.vstack(X_synth_list)
    y_synth = np.concatenate(y_synth_list)

    # shuffle synthetic set
    perm = rng.permutation(len(y_synth))
    return X_synth[perm], y_synth[perm]

# 1) Create 3,000 synthetic rows in selected, normalized feature space 
X_synth, y_synth = generate_gmm_synthetic(
    X_train_selected, y_train,
    n_samples_total=3000,
    n_components=3,
    random_state=42
)

X_synth_df = pd.DataFrame(X_synth, columns=selected_features)
y_synth_ser = pd.Series(y_synth, name="Dementia")

print("\n Synthetic data created:")
print("Synthetic X shape:", X_synth_df.shape, "| Synthetic y shape:", y_synth_ser.shape)
print("Synthetic class counts:\n", y_synth_ser.value_counts())


# 60% synthetic + 40% real


# 1) CREATE MIXED TRAINING DATA (60% synthetic + 40% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.60 * n_real_train)
n_real_used  = int(0.40 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 60% synthetic + 40% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 60% synthetic + 40% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (60% synth + 40% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (60% synth + 40% real)", nb_mixed, X_val_selected, y_val)


# 70% synthetic + 30% real


# 1) CREATE MIXED TRAINING DATA (70% synthetic + 30% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.70 * n_real_train)
n_real_used  = int(0.30 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 70% synthetic + 30% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 70% synthetic + 30% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (70% synthetic + 30% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (70% synthetic + 30% real)", nb_mixed, X_val_selected, y_val)


# 80% Synthetic + 20% real


# 1) CREATE MIXED TRAINING DATA (80% synthetic + 20% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.80 * n_real_train)
n_real_used  = int(0.20 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 80% synthetic + 20% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 80% synthetic + 20% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (80% synthetic + 20% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (80% synthetic + 20% real)", nb_mixed, X_val_selected, y_val)


# 90% synthetic + 10% real


# 1) CREATE MIXED TRAINING DATA (90% synthetic + 10% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.90 * n_real_train)
n_real_used  = int(0.10 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 90% synthetic + 10% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 90% synthetic + 10% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (90% synthetic + 10% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (90% synthetic + 10% real)", nb_mixed, X_val_selected, y_val)


# 100% synthetic + 0% real


# 1) CREATE MIXED TRAINING DATA (100% synthetic + 0% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.100 * n_real_train)
n_real_used  = int(0.0 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 100% synthetic + 0% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 100% synthetic + 0% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (100% synthetic + 0% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (100% synthetic + 0% real)", nb_mixed, X_val_selected, y_val)


# 0% synthetic + 100% real


# 1) CREATE MIXED TRAINING DATA (100% synthetic + 0% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.100 * n_real_train)
n_real_used  = int(0.0 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 100% synthetic + 0% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 100% synthetic + 0% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (100% synthetic + 0% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (100% synthetic + 0% real)", nb_mixed, X_val_selected, y_val)


# 10% synthetic + 90% real


# 1) CREATE MIXED TRAINING DATA (10% synthetic + 90% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.01 * n_real_train)
n_real_used  = int(0.90 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 100% synthetic + 0% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 10% synthetic + 90% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (10% synthetic + 90% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (100% synthetic + 0% real)", nb_mixed, X_val_selected, y_val)


# 20% synthetic + 80% real


# 1) CREATE MIXED TRAINING DATA (100% synthetic + 0% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.20 * n_real_train)
n_real_used  = int(0.80 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 100% synthetic + 0% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 20% synthetic + 80% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (20% synthetic + 80% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (20% synthetic + 80% real)", nb_mixed, X_val_selected, y_val)


# 30% synthetic + 70% real


# 1) CREATE MIXED TRAINING DATA (100% synthetic + 0% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.30 * n_real_train)
n_real_used  = int(0.70 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 100% synthetic + 0% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 30% synthetic + 70% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (30% synthetic + 70% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (30% synthetic + 70% real)", nb_mixed, X_val_selected, y_val)


# 40% synthetic + 60% real


# 1) CREATE MIXED TRAINING DATA (100% synthetic + 0% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.40 * n_real_train)
n_real_used  = int(0.60 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 100% synthetic + 0% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 40% synthetic + 60% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (40% synthetic + 60% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (40% synthetic + 60% real)", nb_mixed, X_val_selected, y_val)


# 50% synthetic + 50% real


# 1) CREATE MIXED TRAINING DATA (100% synthetic + 0% real)

# Convert real training data to DataFrame (must match synthetic format)
X_train_real_df = pd.DataFrame(X_train_selected, columns=selected_features)
y_train_real_ser = y_train.reset_index(drop=True)

# Number of samples to use
n_real_train = len(X_train_real_df)

n_synth_used = int(0.50 * n_real_train)
n_real_used  = int(0.50 * n_real_train)

print("Real training available:", n_real_train)
print("Using real:", n_real_used, "| Using synthetic:", n_synth_used)

# Take random samples from synthetic and real
X_synth_sampled = X_synth_df.sample(n=n_synth_used, random_state=42)
y_synth_sampled = y_synth_ser.sample(n=n_synth_used, random_state=42)

X_real_sampled = X_train_real_df.sample(n=n_real_used, random_state=42)
y_real_sampled = y_train_real_ser.sample(n=n_real_used, random_state=42)

# Combine 50% synthetic + 50% real
X_train_mixed = pd.concat([X_synth_sampled, X_real_sampled], axis=0).reset_index(drop=True)
y_train_mixed = pd.concat([y_synth_sampled, y_real_sampled], axis=0).reset_index(drop=True)

print("\nFinal Mixed Training Set:")
print("X_train_mixed shape:", X_train_mixed.shape)
print("y_train_mixed shape:", y_train_mixed.shape)
print("Class balance:\n", y_train_mixed.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg_mixed = LogisticRegression(max_iter=2000)
logreg_mixed.fit(X_train_mixed, y_train_mixed)

# Naive Bayes
nb_mixed = GaussianNB()
nb_mixed.fit(X_train_mixed, y_train_mixed)

print("\nModels trained on 50% synthetic + 50% real data.")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def eval_model(name, model, X_real, y_real):
    y_pred = model.predict(X_real)
    acc = accuracy_score(y_real, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_real)[:, 1]
            auc = roc_auc_score(y_real, y_proba)
        except:
            pass

    print(f"\n {name} — Evaluation on REAL VALIDATION SET")
    print("Accuracy:", f"{acc:.4f}")
    if auc is not None:
        print("ROC-AUC:", f"{auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_real, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred))


# Evaluate
eval_model("Logistic Regression (50% synthetic + 50% real)", logreg_mixed, X_val_selected, y_val)
eval_model("Naive Bayes (50% synthetic + 50% real)", nb_mixed, X_val_selected, y_val)


# XAI


get_ipython().system('pip install shap lime --quiet')


import numpy as np
import pandas as pd

# convert selected arrays to DataFrames (if they're not already)
if not isinstance(X_train_selected, pd.DataFrame):
    X_train_sel_df = pd.DataFrame(X_train_selected, columns=list(selected_features))
else:
    X_train_sel_df = X_train_selected.copy()

if not isinstance(X_val_selected, pd.DataFrame):
    X_val_sel_df = pd.DataFrame(X_val_selected, columns=list(selected_features))
else:
    X_val_sel_df = X_val_selected.copy()

if not isinstance(X_test_selected, pd.DataFrame):
    X_test_sel_df = pd.DataFrame(X_test_selected, columns=list(selected_features))
else:
    X_test_sel_df = X_test_selected.copy()

# also ensure X_train_mixed is a DataFrame (it already is in your code)
X_train_mixed_df = X_train_mixed.copy() if isinstance(X_train_mixed, pd.DataFrame) else pd.DataFrame(X_train_mixed, columns=list(selected_features))


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def permutation_importances_and_plot(model, X, y, n_repeats=30, random_state=42, title="Permutation Importances"):
    print(f"\nPermutation importance for: {model.__class__.__name__}")
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)
    print(perm_df.head(15))

    plt.figure(figsize=(8,6))
    plt.barh(perm_df['feature'].iloc[::-1], perm_df['importance_mean'].iloc[::-1])
    plt.xlabel("Mean decrease in score (permutation importance)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return perm_df

perm_logreg = permutation_importances_and_plot(logreg_mixed, X_val_sel_df, y_val, title="Permutation Importances — Logistic Regression")
perm_nb     = permutation_importances_and_plot(nb_mixed, X_val_sel_df, y_val, title="Permutation Importances — Naive Bayes")


from sklearn.inspection import PartialDependenceDisplay

# choose top features from permutation importances or RF importances
top_feats = perm_logreg.sort_values("importance_mean", ascending=False).feature.tolist()[:3]

print("PDP features:", top_feats)
for feat in top_feats:
    fig, ax = plt.subplots(figsize=(6,4))
    PartialDependenceDisplay.from_estimator(logreg_mixed, X_val_sel_df, [feat], ax=ax)
    plt.title(f"PDP for {feat} (LogReg)")
    plt.tight_layout()
    plt.show()


import shap
shap.initjs()

# 1) SHAP for Logistic Regression (fast, exact-ish for linear model)
explainer_log = shap.LinearExplainer(logreg_mixed, X_train_mixed_df, feature_perturbation="interventional")
# compute SHAP values for validation
shap_vals_log = explainer_log.shap_values(X_val_sel_df)  # returns array shape (n_samples, n_features)
# summary plot (global)
shap.summary_plot(shap_vals_log, X_val_sel_df, feature_names=X_val_sel_df.columns, show=True)

# 2) SHAP for Naive Bayes using KernelExplainer (slower) - use small background
bg = X_train_mixed_df.sample(n=min(200, len(X_train_mixed_df)), random_state=42)
explainer_nb = shap.KernelExplainer(nb_mixed.predict_proba, bg)
# pick subset of validation to speed up (e.g., first 100)
subset = X_val_sel_df.sample(n=min(100, len(X_val_sel_df)), random_state=42)
shap_vals_nb = explainer_nb.shap_values(subset, nsamples=200)  # list with two arrays for binary
# shap summary for class 1
shap.summary_plot(shap_vals_nb[1], subset, feature_names=subset.columns, show=True)


from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# prepare for lime: training data as numpy
train_for_lime = X_train_mixed_df.values
feature_names = X_train_mixed_df.columns.tolist()
class_names = ['NoDementia','Dementia']  # adapt order to your label encoding

lime_explainer = LimeTabularExplainer(
    training_data=train_for_lime,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

# explain one instance from validation
instance_idx = 0
exp = lime_explainer.explain_instance(
    X_val_sel_df.iloc[instance_idx].values,
    logreg_mixed.predict_proba,
    num_features=6
)
print("LIME explanation for Logistic Regression (top features):")
print(exp.as_list())
exp.show_in_notebook(show_table=True)


# 2D partial dependence between top 2 features
if len(top_feats) >= 2:
    PartialDependenceDisplay.from_estimator(logreg_mixed, X_val_sel_df, [(top_feats[0], top_feats[1])], kind="average")
    plt.title(f"2D PDP: {top_feats[0]} vs {top_feats[1]}")
    plt.show()


perm_logreg.to_csv("perm_importance_logreg.csv", index=False)
perm_nb.to_csv("perm_importance_nb.csv", index=False)
# save shap values (small subset) as npz
np.savez_compressed("shap_logreg_vals.npz", shap_vals_log=shap_vals_log)




