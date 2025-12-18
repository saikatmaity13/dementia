import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. RELOAD YOUR DATA
# ==========================================
# REPLACE 'your_dataset.csv' with the actual name of your file (e.g., 'features_30sec.csv' or similar)
# If you used a combined synthetic+real dataset, load that one.
df = pd.read_csv("dementia_patients_health_data.csv") 

# Define your features (X) and target (y)
# Adjust 'label' if your target column has a different name
X = df.drop(['label', 'filename'], axis=1, errors='ignore') 
y = df['label']

# ==========================================
# 2. RECREATE X_train
# ==========================================
# This creates the variable 'X_train' that was missing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. DEFINE AND FIT THE SCALER
# ==========================================
scaler = MinMaxScaler()
scaler.fit(X_train)  # Now X_train exists, so this will work!

# ==========================================
# 4. RECREATE THE MODEL (Optional if you lost logreg_mixed too)
# ==========================================
# If 'logreg_mixed' is also missing, we need to retrain it quickly:
logreg_mixed = LogisticRegression(max_iter=1000)
logreg_mixed.fit(scaler.transform(X_train), y_train)

# ==========================================
# 5. SAVE EVERYTHING
# ==========================================
# Save Scaler
joblib.dump(scaler, 'minmax_scaler.pkl')

# Save Model
joblib.dump(logreg_mixed, 'logistic_regression_model.pkl')

# Save Feature Names
# We save the column names from X so the app knows what features to expect
joblib.dump(X.columns.tolist(), 'selected_features.pkl')

print("SUCCESS: X_train recreated. All 3 files (scaler, model, features) are saved.")