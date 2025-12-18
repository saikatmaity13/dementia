import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
from lime.lime_tabular import LimeTabularExplainer

# Page Config
st.set_page_config(page_title="Dementia Prediction System", layout="wide")

st.title("🧠 Dementia Prediction & Synthetic AI System")
st.markdown("""
**System Architecture:**
1. **Prediction Calculator:** Real-time dementia risk assessment for new patients.
2. **Synthetic Experiment:** Research sandbox to train models with Synthetic Data (GMM).
3. **Explainability:** AI transparency tools (SHAP/LIME) to understand the logic.
""")

# --- GLOBAL HELPER FUNCTIONS & CACHING ---

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def clean_and_encode_data(df):
    """
    Cleans data and handles encoding. 
    Returns:
    - data_encoded: The numeric dataframe ready for ML
    - encoders: Dictionary of label encoders (for reversing transformations)
    - input_structure: Dictionary for building the UI form
    - feature_cols: List of final column names
    """
    data = df.copy()
    
    # 1. UI Structure (Capture before encoding)
    input_structure = {}
    if 'Dementia' in data.columns:
        cols_to_scan = data.drop('Dementia', axis=1).columns
    else:
        cols_to_scan = data.columns
        
    for col in cols_to_scan:
        if data[col].dtype == 'object':
            input_structure[col] = list(data[col].dropna().unique())
        else:
            input_structure[col] = 'numeric'

    # 2. Impute
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

    # 3. Drop Duplicates
    data.drop_duplicates(inplace=True)

    # 4. Outlier Handling (IQR)
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower, upper)

    # 5. Encoding
    encoders = {}
    label_cols = ['Education_Level', 'Physical_Activity']
    for col in label_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le
            
    # One-Hot Encoding
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    feature_cols = data_encoded.drop('Dementia', axis=1).columns if 'Dementia' in data_encoded.columns else data_encoded.columns

    return data_encoded, encoders, input_structure, feature_cols, categorical_cols.tolist()

@st.cache_data
def perform_feature_selection(X_train_scaled, y_train, _feature_names):
    # RF Selection
    rf_selector = RandomForestClassifier(random_state=42)
    rf_selector.fit(X_train_scaled, y_train)
    
    selector = SelectFromModel(rf_selector, threshold=0.02, prefit=True)
    support = selector.get_support()
    selected_feats = _feature_names[support]
    
    return selector, selected_feats

def generate_gmm_synthetic(X_train_sel, y_train, n_samples_total=3000, n_components=3):
    rng = np.random.RandomState(42)
    classes, counts = np.unique(y_train, return_counts=True)
    class_probs = counts / counts.sum()
    
    n_per_class = np.floor(class_probs * n_samples_total).astype(int)
    remainder = n_samples_total - n_per_class.sum()
    
    # Distribute remainder
    residuals = (class_probs * n_samples_total) - n_per_class
    for idx in np.argsort(residuals)[::-1][:remainder]:
        n_per_class[idx] += 1
        
    X_synth_list, y_synth_list = [], []
    
    for c, n_c in zip(classes, n_per_class):
        X_c = X_train_sel[y_train.values == c]
        gmm = GaussianMixture(n_components=n_components, covariance_type="full", reg_covar=1e-6, random_state=42)
        gmm.fit(X_c)
        X_c_synth, _ = gmm.sample(n_c)
        X_c_synth = np.clip(X_c_synth, 0.0, 1.0)
        X_synth_list.append(X_c_synth)
        y_synth_list.append(np.full(n_c, c, dtype=int))
        
    X_synth = np.vstack(X_synth_list)
    y_synth = np.concatenate(y_synth_list)
    perm = rng.permutation(len(y_synth))
    return X_synth[perm], y_synth[perm]

# --- MAIN APP LOGIC ---

st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    # 1. LOAD & INITIAL PROCESS
    raw_df = load_data(uploaded_file)
    
    if 'Dementia' not in raw_df.columns:
        st.error("Dataset must contain 'Dementia' column.")
    else:
        # 2. RUN PIPELINE ONCE
        data_encoded, encoders, input_structure, feature_cols, cat_cols = clean_and_encode_data(raw_df)
        
        X = data_encoded.drop('Dementia', axis=1)
        y = data_encoded['Dementia']
        
        # Split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Scale
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Feature Selection
        selector, selected_feats = perform_feature_selection(X_train_scaled, y_train, feature_cols)
        
        # Transform to selected features
        X_train_sel = selector.transform(X_train_scaled)
        X_val_sel = selector.transform(X_val_scaled)
        
        # Train Baseline Model (for Tab 1 Calculator) on Real Data
        base_model = LogisticRegression(max_iter=1000)
        base_model.fit(X_train_sel, y_train)

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🔮 Prediction Calculator", "🧪 Synthetic Experiment", "🔍 Explainability"])

        # ==========================================
        # TAB 1: PREDICTION CALCULATOR
        # ==========================================
        with tab1:
            st.subheader("Patient Risk Calculator")
            st.info("This calculator uses a model trained on 100% Real Data.")
            
            with st.form("prediction_form"):
                input_data = {}
                cols_ui = st.columns(3)
                idx = 0
                
                for feature_name, feature_type in input_structure.items():
                    col = cols_ui[idx % 3]
                    if feature_type == 'numeric':
                        input_data[feature_name] = col.number_input(f"{feature_name}", value=0.0)
                    else:
                        input_data[feature_name] = col.selectbox(f"{feature_name}", feature_type)
                    idx += 1
                
                submitted = st.form_submit_button("Predict Condition")
                
            if submitted:
                # Process Input
                input_df = pd.DataFrame([input_data])
                
                # Label Encode
                for col, le in encoders.items():
                    if col in input_df.columns:
                        try:
                            input_df[col] = le.transform(input_df[col])
                        except:
                            st.error(f"Unknown value in {col}")

                # One-Hot Encode
                input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
                
                # Align Columns (Fill missing with 0)
                input_df = input_df.reindex(columns=feature_cols, fill_value=0)
                
                # Scale
                input_scaled = scaler.transform(input_df)
                
                # Select Features
                input_final = selector.transform(input_scaled)
                
                # Predict
                pred = base_model.predict(input_final)[0]
                prob = base_model.predict_proba(input_final)[0][1]
                
                st.divider()
                if pred == 1:
                    st.error(f"## Dementia Detected (Confidence: {prob:.2%})")
                    st.warning("Recommendation: Consult a specialist.")
                else:
                    st.success(f"## Non-Dementia (Confidence: {(1-prob):.2%})")
                    st.info("Recommendation: Routine checkups.")

        # ==========================================
        # TAB 2: SYNTHETIC EXPERIMENT
        # ==========================================
        with tab2:
            st.subheader("Train on Mixed Data (Real + Synthetic)")
            st.markdown("Use the slider to augment the real training data with GMM-generated synthetic samples.")
            
            col_slide, col_btn = st.columns([3, 1])
            with col_slide:
                synth_ratio = st.slider("Synthetic Data Percentage", 0, 100, 50, step=10)
            with col_btn:
                st.write("") # Spacer
                st.write("") 
                train_btn = st.button("Train Mixed Models")
                
            if train_btn:
                with st.spinner("Generating Synthetic Data & Training..."):
                    # 1. Generate Synthetic
                    X_synth, y_synth = generate_gmm_synthetic(X_train_sel, y_train)
                    
                    # 2. Mix Logic
                    # n_real_total = len(X_train_sel)
                    # For simplicity: Keep total training size roughly consistent or additive
                    # Strategy: Take Ratio% synthetic and (1-Ratio)% real
                    
                    X_train_df = pd.DataFrame(X_train_sel, columns=selected_feats)
                    y_train_ser = y_train.reset_index(drop=True)
                    X_synth_df = pd.DataFrame(X_synth, columns=selected_feats)
                    y_synth_ser = pd.Series(y_synth, name="Dementia")
                    
                    n_total = len(X_train_df)
                    n_synth = int(n_total * (synth_ratio / 100))
                    n_real = int(n_total * ((100 - synth_ratio) / 100))
                    
                    if n_synth > 0:
                        X_s = X_synth_df.sample(n=n_synth, replace=True, random_state=42)
                        y_s = y_synth_ser.sample(n=n_synth, replace=True, random_state=42)
                    else:
                        X_s = pd.DataFrame(columns=selected_feats)
                        y_s = pd.Series(dtype='int')
                        
                    if n_real > 0:
                        X_r = X_train_df.sample(n=n_real, replace=False, random_state=42)
                        y_r = y_train_ser.sample(n=n_real, replace=False, random_state=42)
                    else:
                        X_r = pd.DataFrame(columns=selected_feats)
                        y_r = pd.Series(dtype='int')
                        
                    X_mixed = pd.concat([X_s, X_r], axis=0)
                    y_mixed = pd.concat([y_s, y_r], axis=0)
                    
                    # 3. Train Models
                    log_mixed = LogisticRegression(max_iter=1000)
                    log_mixed.fit(X_mixed, y_mixed)
                    
                    nb_mixed = GaussianNB()
                    nb_mixed.fit(X_mixed, y_mixed)
                    
                    # 4. Evaluate
                    y_pred_log = log_mixed.predict(X_val_sel)
                    y_pred_nb = nb_mixed.predict(X_val_sel)
                    
                    acc_log = accuracy_score(y_val, y_pred_log)
                    acc_nb = accuracy_score(y_val, y_pred_nb)
                    
                    # Display Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("LogReg Accuracy (Mixed)", f"{acc_log:.4f}")
                    m2.metric("Naive Bayes Accuracy (Mixed)", f"{acc_nb:.4f}")
                    
                    st.write("Confusion Matrix (LogReg):")
                    st.write(confusion_matrix(y_val, y_pred_log))
                    
                    # Save for Tab 3
                    st.session_state['mixed_model'] = log_mixed
                    st.session_state['X_mixed'] = X_mixed
                    st.session_state['exp_run'] = True
                    st.success("Models updated! Go to the 'Explainability' tab to visualize.")

        # ==========================================
        # TAB 3: EXPLAINABILITY (XAI)
        # ==========================================
        with tab3:
            st.subheader("Explainable AI (XAI)")
            
            # Check if experiment run, else use base model
            if 'exp_run' in st.session_state and st.session_state['exp_run']:
                model_to_explain = st.session_state['mixed_model']
                data_to_explain = st.session_state['X_mixed']
                st.caption("Explaining the **Mixed Model** (from Synthetic Experiment tab).")
            else:
                model_to_explain = base_model
                data_to_explain = pd.DataFrame(X_train_sel, columns=selected_feats)
                st.caption("Explaining the **Base Model** (Real Data only). Run the Experiment tab to explain mixed models.")
            
            # SHAP
            st.markdown("### 1. Global Importance (SHAP)")
            if st.button("Run SHAP Analysis"):
                with st.spinner("Calculating SHAP values..."):
                    explainer = shap.LinearExplainer(model_to_explain, data_to_explain)
                    X_val_df = pd.DataFrame(X_val_sel, columns=selected_feats)
                    shap_values = explainer.shap_values(X_val_df)
                    
                    fig_shap, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_val_df, show=False)
                    st.pyplot(fig_shap)
            
            # LIME
            st.markdown("### 2. Local Explanation (LIME)")
            st.write("Explain why a specific validation row was classified this way.")
            idx_to_explain = st.number_input("Validation Row Index", 0, len(X_val_sel)-1, 0)
            
            if st.button("Run LIME"):
                explainer_lime = LimeTabularExplainer(
                    training_data=data_to_explain.values,
                    feature_names=selected_feats,
                    class_names=['NoDementia', 'Dementia'],
                    mode='classification'
                )
                exp = explainer_lime.explain_instance(
                    X_val_sel[idx_to_explain],
                    model_to_explain.predict_proba,
                    num_features=6
                )
                fig_lime = exp.as_pyplot_figure()
                st.pyplot(fig_lime)

else:
    st.info("⬅️ Please upload 'dementia_patients_health_data.csv' to start.")