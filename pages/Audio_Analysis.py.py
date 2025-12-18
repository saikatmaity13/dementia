import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import librosa
import librosa.display
import whisper
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import graphviz

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dementia Research Lab", layout="wide", page_icon="🧠")
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. GLOBAL SETUP & CACHING ---

@st.cache_resource
def load_saved_models():
    try:
        text_model = joblib.load('models/text_model.pkl')
        audio_model = joblib.load('models/audio_model.pkl')
        audio_features = joblib.load('models/audio_features.pkl')
        return text_model, audio_model, audio_features
    except FileNotFoundError:
        return None, None, None

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords.zip')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
    return set(stopwords.words('english')), WordNetLemmatizer()

@st.cache_resource
def load_mlm_model():
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")
    return tokenizer, model

# Load resources
text_model, audio_model, audio_cols = load_saved_models()
whisper_model = load_whisper()
stop_words, lemmatizer = setup_nltk()
mlm_tokenizer, mlm_model = load_mlm_model()

# --- 2. HELPER FUNCTIONS ---

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    processed = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(processed)

def extract_audio_features_single(audio_path, transcript_text):
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    rms = np.mean(librosa.feature.rms(y=y))
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    zcr_std = np.std(librosa.feature.zero_crossing_rate(y))
    
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    valid_f0 = f0[~np.isnan(f0)]
    pitch_std = np.std(valid_f0) if len(valid_f0) > 0 else 0
    
    non_silent = librosa.effects.split(y, top_db=20)
    n_pauses = max(0, len(non_silent) - 1)
    pause_rate = n_pauses / duration if duration > 0 else 0
    
    word_count = len(transcript_text.split())
    speech_rate = word_count / duration if duration > 0 else 0
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    features = {
        'duration': duration, 'rms_mean': rms, 'zcr_mean': zcr_mean, 'zcr_std': zcr_std,
        'pitch_std': pitch_std, 'pause_rate': pause_rate, 'speech_rate': speech_rate
    }
    for i, val in enumerate(mfccs_mean):
        features[f'mfcc_{i+1}'] = val
    return features

def generate_synthetic_sentence(text, tokenizer, model, num_masks=1):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids'][0]
    indices_to_mask = random.sample(range(1, len(input_ids)-1), min(num_masks, len(input_ids)-2))
    input_ids[indices_to_mask] = tokenizer.mask_token_id
    
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
        predictions = outputs.logits[0]
    
    for idx in indices_to_mask:
        predicted_token_id = torch.argmax(predictions[idx]).item()
        input_ids[idx] = predicted_token_id
        
    return tokenizer.decode(input_ids, skip_special_tokens=True)

# --- 3. APP INTERFACE ---

st.title(" Dementia Research Platform")
st.markdown("### A Multimodal AI System for Early Detection")

# CREATE 5 TABS
tab1, tab2, tab3, tab4,  = st.tabs([
    " Patient Diagnosis", 
    " Synthetic Data Experiment", 
    " Generate New Data", 
    " Manual Feature Input",
    
])

# ==========================================
# TAB 1: DIAGNOSIS (ADVANCED VISUALS)
# ==========================================
with tab1:
    st.header("Patient Diagnosis (Audio File)")
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_file = st.file_uploader("Upload .wav file", type=['wav'])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        with col2:
            st.audio(uploaded_file)
            
            with st.expander("🔊 Signal Processing Analysis (Spectrogram)", expanded=False):
                y, sr = librosa.load(tmp_path, sr=None)
                fig_audio, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='blue', alpha=0.6)
                ax1.set(title='Amplitude Waveform', ylabel='Amplitude')
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2, cmap='magma')
                ax2.set(title='Mel-Spectrogram', ylabel='Hz')
                fig_audio.colorbar(img, ax=ax2, format="%+2.0f dB")
                st.pyplot(fig_audio)

            if st.button("Run Advanced Diagnosis"):
                if text_model is None:
                    st.error("Models missing! Run 'save_models.py'.")
                else:
                    with st.spinner("Analyzing linguistic and acoustic markers..."):
                        res = whisper_model.transcribe(tmp_path)
                        text = res['text']
                        clean_txt = clean_text(text)
                        feats = extract_audio_features_single(tmp_path, text)
                        
                        df_f = pd.DataFrame([feats])
                        for c in audio_cols: 
                            if c not in df_f: df_f[c]=0
                        df_f = df_f[audio_cols]
                        
                        p_text = text_model.predict_proba([clean_txt])[0]
                        p_audio = audio_model.predict_proba(df_f)[0]
                        
                        final_p_dem = (p_text[0]*2 + p_audio[0]*1)/3
                        final_p_nodem = (p_text[1]*2 + p_audio[1]*1)/3
                        
                        pred = "Dementia" if final_p_dem > final_p_nodem else "No Dementia"
                        conf = max(final_p_dem, final_p_nodem)
                        
                        st.divider()
                        
                        c_res1, c_res2 = st.columns([1, 1.5])
                        with c_res1:
                            st.subheader("Diagnosis")
                            gauge_val = conf * 100 if pred == "Dementia" else (1-conf)*100
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = gauge_val,
                                title = {'text': "Dementia Probability Risk"},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps' : [{'range': [0, 50], 'color': "lightgreen"}, {'range': [50, 100], 'color': "salmon"}],
                                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                                }
                            ))
                            fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20))
                            st.plotly_chart(fig_gauge, use_container_width=True)
                            st.caption("Transcription:")
                            st.info(f'"{text}"')

                        with c_res2:
                            st.subheader(" Patient vs. Baselines")
                            categories = ['Speech Rate', 'Pause Rate', 'Pitch Std', 'ZCR Mean', 'Energy']
                            raw_healthy = [2.6, 0.4, 35.0, 0.1, 0.06]
                            raw_dementia = [1.8, 0.8, 15.0, 0.15, 0.03]
                            raw_patient = [feats['speech_rate'], feats['pause_rate'], feats['pitch_std'], feats['zcr_mean'], feats['rms_mean']]

                            norm_healthy = [100, 100, 100, 100, 100]
                            norm_dementia = [(d/h)*100 if h!=0 else 0 for d, h in zip(raw_dementia, raw_healthy)]
                            norm_patient = [(p/h)*100 if h!=0 else 0 for p, h in zip(raw_patient, raw_healthy)]

                            fig_bar = go.Figure()
                            fig_bar.add_trace(go.Bar(name='Avg Dementia', x=categories, y=norm_dementia, marker_color='salmon', opacity=0.7))
                            fig_bar.add_trace(go.Bar(name='Avg Healthy', x=categories, y=norm_healthy, marker_color='lightgreen', opacity=0.7))
                            fig_bar.add_trace(go.Bar(name='Current Patient', x=categories, y=norm_patient, marker_color='#4F8BF9', text=[f"{v:.0f}%" for v in norm_patient], textposition='auto'))

                            fig_bar.update_layout(barmode='group', height=350, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="% of Healthy Baseline", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            st.plotly_chart(fig_bar, use_container_width=True)

                        st.divider()
                        st.subheader(" Explainable AI: Linguistic Analysis")
                        st.write("Words contributing most to the diagnosis (Red = Dementia Risk, Green = Healthy)")
                        
                        feature_names = text_model.named_steps['tfidf'].get_feature_names_out()
                        coefs = text_model.named_steps['clf'].coef_[0]
                        word_impact = {}
                        for word in set(clean_txt.split()):
                            if word in feature_names:
                                idx = list(feature_names).index(word)
                                score = coefs[idx]
                                if score != 0: word_impact[word] = score
                        
                        sorted_impact = sorted(word_impact.items(), key=lambda x: x[1])
                        
                        if sorted_impact:
                            indicators = sorted_impact[:5] + sorted_impact[-5:]
                            words = [x[0] for x in indicators]
                            scores = [x[1] for x in indicators]
                            colors = ['red' if x < 0 else 'green' for x in scores]
                            
                            fig_xai, ax = plt.subplots(figsize=(10, 4))
                            ax.barh(words, scores, color=colors)
                            ax.axvline(0, color='black', linewidth=0.8)
                            st.pyplot(fig_xai)
                        else:
                            st.info("No strong linguistic markers found in this clip.")
                            
        os.remove(tmp_path)

# ==========================================
# TAB 2: SYNTHETIC EXPERIMENT (COMPLEX)
# ==========================================
with tab2:
    st.header(" Synthetic Data Research Lab")
    st.markdown("""
    **Experiment:** Determine the optimal ratio of Synthetic-to-Real data. 
    Does blending AI-generated transcripts improve model generalization?
    """)
    
    try:
        df_real = pd.read_csv("transcripts_with_ids.csv")
        df_synth = pd.read_csv("augmented_data/transcripts_augmented_mlm.csv")
        
        df_merged = pd.merge(df_synth, df_real, on=['Processed_Text', 'Diagnosis'], how='left', indicator=True)
        df_synth_only = df_merged[df_merged['_merge'] == 'left_only'][['Processed_Text', 'Diagnosis']]
        
        df_real['Source'] = 'Real Patient'
        df_synth_only['Source'] = 'Synthetic AI'
        
        st.subheader(" Experiment Controls")
        c_ctrl1, c_ctrl2 = st.columns(2)
        
        max_available = len(df_real) + len(df_synth_only)
        
        with c_ctrl1:
            total_samples = st.slider("Total Training Set Size", 100, max_available, min(1000, max_available), step=50)
        with c_ctrl2:
            mix_ratio = st.slider("Synthetic Data Percentage (%)", 0, 100, 30)

        n_synth = int(total_samples * (mix_ratio / 100))
        n_real = total_samples - n_synth
        
        if n_real > len(df_real): n_real = len(df_real)
        if n_synth > len(df_synth_only): n_synth = len(df_synth_only)
        
        df_vis_real = df_real.sample(n=n_real, replace=True, random_state=42)
        df_vis_synth = df_synth_only.sample(n=n_synth, replace=True, random_state=42) if n_synth > 0 else pd.DataFrame()
        df_experiment = pd.concat([df_vis_real, df_vis_synth])
        
        with st.expander(" View Data Manifold (PCA Projection)", expanded=True):
            if len(df_experiment) > 10:
                tfidf_vis = TfidfVectorizer(max_features=300)
                vectors = tfidf_vis.fit_transform(df_experiment['Processed_Text'].fillna(""))
                pca = PCA(n_components=2)
                coords = pca.fit_transform(vectors.toarray())
                df_experiment['x'] = coords[:, 0]
                df_experiment['y'] = coords[:, 1]
                
                fig_pca = px.scatter(df_experiment, x='x', y='y', color='Source', symbol='Diagnosis', opacity=0.7, title="Semantic Distribution", color_discrete_map={'Real Patient': '#4F8BF9', 'Synthetic AI': '#FF6B6B'})
                fig_pca.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_pca, use_container_width=True)

        if st.button(" Run Training Simulation", type="primary"):
            my_bar = st.progress(0, text="Training models...")
            
            df_real_train_pool, df_real_test = train_test_split(df_real, test_size=0.20, random_state=42, stratify=df_real['Diagnosis'])
            my_bar.progress(30, text="Prepared Test Set...")

            if n_real > len(df_real_train_pool): n_real = len(df_real_train_pool)
            df_train_real = df_real_train_pool.sample(n=n_real, replace=False, random_state=42)
            
            if n_synth > 0:
                df_train_synth = df_synth_only.sample(n=n_synth, replace=True, random_state=42)
                df_train_final = pd.concat([df_train_real, df_train_synth])
            else:
                df_train_final = df_train_real
                
            my_bar.progress(60, text="Vectorizing Text...")

            pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=5000)), ('clf', LogisticRegression(class_weight='balanced', solver='liblinear'))])
            pipeline.fit(df_train_final['Processed_Text'].fillna(""), df_train_final['Diagnosis'])
            my_bar.progress(90, text="Evaluating...")

            preds = pipeline.predict(df_real_test['Processed_Text'].fillna(""))
            current_acc = accuracy_score(df_real_test['Diagnosis'], preds)
            my_bar.progress(100, text="Complete!")
            
            st.divider()
            BASELINE_ACC = 0.7610 
            delta = current_acc - BASELINE_ACC
            
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1: st.metric(label="Current Mix Accuracy", value=f"{current_acc:.2%}", delta=f"{delta:.2%}", delta_color="normal")
            with c_r2: st.metric(label="Original Baseline", value="76.10%")
            with c_r3: st.metric(label="Data Composition", value=f"{mix_ratio}% Synthetic")

            st.subheader(" Performance Comparison")
            chart_data = pd.DataFrame({"Model Configuration": ["Original Baseline (Real Only)", f"Current Experiment ({mix_ratio}% Synth)"], "Accuracy": [BASELINE_ACC, current_acc]})
            fig_comp = go.Figure(go.Bar(x=chart_data["Accuracy"], y=chart_data["Model Configuration"], orientation='h', marker_color=['lightgray', '#4F8BF9'], text=[f"{v:.1%}" for v in chart_data["Accuracy"]], textposition='auto'))
            fig_comp.update_layout(xaxis=dict(range=[0.5, 1.0], title="Accuracy"), height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_comp, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading experiment data: {e}")

# ==========================================
# TAB 3: GENERATE DATA
# ==========================================
with tab3:
    st.header("🧬 AI Data Generator")
    c_ctrl1, c_ctrl2 = st.columns([1, 2])
    with c_ctrl1:
        st.subheader(" Configuration")
        target_class = st.selectbox("Target Diagnosis Class", ["dementia", "nodementia"])
        input_method = st.radio("Input Source", ["Use Random Seed from Data", "Type My Own Sentence"])
        with st.expander(" Model Hyperparameters", expanded=True):
            mask_prob = st.slider("Mutation Rate (Mask %)", 0.1, 0.5, 0.15)
            top_k = st.slider("Creativity (Top-K)", 1, 50, 10)
            n_gen = st.number_input("Variations to Generate", 1, 10, 3)

    with c_ctrl2:
        st.subheader(" Input Data")
        seed_text = ""
        if input_method == "Type My Own Sentence":
            seed_text = st.text_area("Enter a base sentence:", "The quick brown fox jumps over the lazy dog.")
        else:
            try:
                df_real = pd.read_csv("transcripts_with_ids.csv")
                possible_seeds = df_real[df_real['Diagnosis'] == target_class]['Processed_Text'].dropna().tolist()
                if st.button(" Roll Dice for Random Seed"):
                    st.session_state['current_seed'] = random.choice(possible_seeds)
                if 'current_seed' in st.session_state:
                    seed_text = st.session_state['current_seed']
                    st.info(f"**Seed:** ...{seed_text[:100]}...")
            except:
                st.error("Data file not found.")

    if st.button(" Generate Variations", type="primary"):
        if not seed_text:
            st.error("Please provide a seed sentence.")
        else:
            st.divider()
            st.subheader(" Generated Mutations")
            def mutate_sentence(text, prob, k):
                inputs = mlm_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                input_ids = inputs['input_ids'][0]
                n_tokens = len(input_ids) - 2 
                n_mask = max(1, int(n_tokens * prob))
                mask_indices = random.sample(range(1, len(input_ids)-1), min(n_mask, n_tokens))
                input_ids[mask_indices] = mlm_tokenizer.mask_token_id
                with torch.no_grad(): logits = mlm_model(input_ids.unsqueeze(0)).logits
                changes = []
                for idx in mask_indices:
                    top_k_logits, top_k_inds = torch.topk(logits[0, idx], k)
                    selected_idx = random.choice(top_k_inds).item()
                    new_word = mlm_tokenizer.decode([selected_idx]).strip()
                    input_ids[idx] = selected_idx
                    changes.append(new_word)
                return mlm_tokenizer.decode(input_ids, skip_special_tokens=True), changes

            for i in range(n_gen):
                new_sent, changed_words = mutate_sentence(seed_text, mask_prob, top_k)
                prob_check = text_model.predict_proba([new_sent])[0]
                is_dementia = prob_check[0] > prob_check[1]
                predicted_label = "dementia" if is_dementia else "nodementia"
                match = (predicted_label == target_class)
                with st.container():
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**Variation {i+1}:**")
                        st.write(new_sent)
                        st.caption(f"Changes: {', '.join(changed_words)}")
                    with c2:
                        if match: st.success(f" Valid\n({target_class})")
                        else: st.warning(f" Drift\n(Scored as {predicted_label})")
                    st.divider()

# ==========================================
# ==========================================
# TAB 4: CLINICAL SIMULATION (AUTO-CALC)
# ==========================================
with tab4:
    st.header(" Clinical Simulation Sandbox")
    st.markdown("Use this tab to **automatically calculate** metrics from a file, then tweak them to simulate scenarios.")
    
    if audio_model is None:
        st.error("Models not found. Run 'save_models.py'.")
    else:
        # Initialize session state for manual inputs
        if 'man_speech' not in st.session_state: 
            st.session_state.update({
                'man_speech': 2.5, 'man_pause': 0.5, 'man_pitch': 20.0, 
                'man_rms': 0.05, 'man_dur': 10.0, 'man_zcr_m': 0.1, 'man_zcr_s': 0.05,
                'man_mfccs': {f'mfcc_{i}': 0.0 for i in range(1, 14)}
            })

        # --- PART 1: AUTO-EXTRACTOR ---
        st.subheader("1. Input Source")
        input_method = st.radio("Choose Input Method:", [" Manual Entry", " Drag & Drop File", " Record Voice"], horizontal=True)
        
        audio_source = None
        
        if input_method == " Drag & Drop File":
            audio_source = st.file_uploader("Upload .wav for auto-calculation", type=['wav'], key="man_uploader")
        elif input_method == " Record Voice":
            # Using standard st.audio_input (Streamlit 1.40+)
            audio_source = st.audio_input("Record your voice")

        # PROCESS AUDIO IF PROVIDED
        if audio_source is not None:
            # We need to process this only when a button is clicked or file changes
            if st.button(" Analyze & Populate Fields"):
                with st.spinner("Transcribing and extracting features..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_source.getvalue())
                        tmp_p = tmp.name
                    
                    # Extraction Logic
                    res = whisper_model.transcribe(tmp_p)
                    feats = extract_audio_features_single(tmp_p, res['text'])
                    os.remove(tmp_p)
                    
                    # Update Session State
                    st.session_state['man_speech'] = float(feats['speech_rate'])
                    st.session_state['man_pause'] = float(feats['pause_rate'])
                    st.session_state['man_pitch'] = float(feats['pitch_std'])
                    st.session_state['man_rms'] = float(feats['rms_mean'])
                    st.session_state['man_dur'] = float(feats['duration'])
                    st.session_state['man_zcr_m'] = float(feats['zcr_mean'])
                    st.session_state['man_zcr_s'] = float(feats['zcr_std'])
                    for i in range(1, 14):
                        st.session_state['man_mfccs'][f'mfcc_{i}'] = float(feats.get(f'mfcc_{i}', 0.0))
                    
                    st.success(" Fields Populated! You can now edit them below.")
                    st.rerun()

        st.divider()

        # --- PART 2: MANUAL INPUT FORM ---
        with st.form("manual_input_form"):
            st.subheader("2. Patient Metrics (Editable)")
            
            with st.container(border=True):
                st.markdown("####  Voice Rhythm")
                c1, c2, c3 = st.columns(3)
                with c1: speech_rate = st.number_input("Speech Rate", 0.0, 20.0, st.session_state['man_speech'], key='man_speech')
                with c2: pause_rate = st.number_input("Pause Rate", 0.0, 20.0, st.session_state['man_pause'], key='man_pause')
                with c3: pitch_std = st.number_input("Pitch Std", 0.0, 1000.0, st.session_state['man_pitch'], key='man_pitch')
            
            with st.container(border=True):
                st.markdown("####  Signal Quality")
                c4, c5 = st.columns(2)
                with c4: 
                    duration = st.number_input("Duration", 0.0, 600.0, value=st.session_state['man_dur'], key='man_dur')
                    rms_mean = st.number_input("Energy (RMS)", 0.0, 10.0, st.session_state['man_rms'], key='man_rms') # increased max
                with c5:
                    zcr_mean = st.number_input("ZCR Mean", value=st.session_state['man_zcr_m'], key='man_zcr_m')
                    zcr_std = st.number_input("ZCR Std", value=st.session_state['man_zcr_s'], key='man_zcr_s')
            
            with st.expander("Advanced MFCCs"):
                mfcc_inputs = {}
                cols = st.columns(4)
                for i in range(1, 14):
                    with cols[(i-1)%4]: 
                        val = st.session_state['man_mfccs'].get(f'mfcc_{i}', 0.0)
                        mfcc_inputs[f'mfcc_{i}'] = st.number_input(f"MFCC {i}", value=val)

            submitted = st.form_submit_button("Run Simulation Prediction", type="primary")

            if submitted:
                input_data = {'duration': duration, 'rms_mean': rms_mean, 'zcr_mean': zcr_mean, 'zcr_std': zcr_std, 'pitch_std': pitch_std, 'pause_rate': pause_rate, 'speech_rate': speech_rate}
                input_data.update(mfcc_inputs)
                df_input = pd.DataFrame([input_data])
                for col in audio_cols: 
                    if col not in df_input.columns: df_input[col] = 0
                df_input = df_input[audio_cols]
                
                probs = audio_model.predict_proba(df_input)[0]
                if 'dementia' in audio_model.classes_: dem_idx = list(audio_model.classes_).index('dementia')
                else: dem_idx = 0 
                risk = probs[dem_idx]
                
                st.divider()
                c_res1, c_res2 = st.columns([1, 1.5])
                with c_res1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number", value = risk * 100, title = {'text': "Dementia Risk"},
                        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}, 'steps' : [{'range': [0, 50], 'color': "lightgreen"}, {'range': [50, 100], 'color': "salmon"}], 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with c_res2:
                    st.subheader(" Simulation: Speech Rate Impact")
                    sim_range = np.linspace(0.5, 4.0, 50)
                    sim_probs = []
                    sim_input = df_input.copy()
                    for val in sim_range:
                        sim_input['speech_rate'] = val
                        sim_probs.append(audio_model.predict_proba(sim_input)[0][dem_idx])
                    
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=sim_range, y=sim_probs, mode='lines', name='Risk Curve', line=dict(color='red', width=3)))
                    fig_sim.add_trace(go.Scatter(x=[speech_rate], y=[risk], mode='markers', marker=dict(color='blue', size=12, symbol='x'), name='Current Value'))
                    fig_sim.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20), xaxis_title="Speech Rate", yaxis_title="Risk Probability")
                    st.plotly_chart(fig_sim, use_container_width=True)


