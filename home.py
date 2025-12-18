import streamlit as st

# Global Page Config
st.set_page_config(
    page_title="Dementia Research Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- MAIN LANDING PAGE ---
st.title("🧠 Multimodal Dementia Research Platform")
st.markdown("### Integrating Acoustic, Linguistic, and Clinical Data")

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.info("### 🎙️ Module 1: Audio Analysis")
    st.write("""
    **Focus:** Detecting dementia markers from speech patterns.
    - **Input:** Raw Audio (.wav)
    - **Features:** Pitch, Pauses, Speech Rate, Vocabulary.
    - **Models:** Whisper + Random Forest + Logistic Regression.
    - **Goal:** Early screening via voice biomarkers.
    """)
    st.success("👈 Select **'Audio Analysis'** in the sidebar to start.")

with c2:
    st.info("### 📋 Module 2: Clinical Records")
    st.write("""
    **Focus:** Risk calculation based on medical history.
    - **Input:** Tabular Patient Data (.csv)
    - **Features:** MRI scores, Age, Education, SES.
    - **Models:** Mixed-Data Training + Explainable AI (SHAP/LIME).
    - **Goal:** Clinical risk stratification.
    """)
    st.success("👈 Select **'Clinical Records'** in the sidebar to start.")

st.divider()

st.subheader("🛠️ System Architecture")
# You can use your Graphviz chart here to show the whole system!
st.graphviz_chart('''
    digraph {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor=white];
        
        User [shape=ellipse, fillcolor="#E3F2FD"];
        
        subgraph cluster_0 {
            label = "Module 1: Audio";
            style=dashed;
            Audio [label="Speech Signal"];
            NLP [label="Linguistic Model"];
        }
        
        subgraph cluster_1 {
            label = "Module 2: Clinical";
            style=dashed;
            CSV [label="Patient Metadata"];
            Tabular [label="Risk Classifier"];
        }
        
        User -> Audio;
        User -> CSV;
        NLP -> Diagnosis;
        Tabular -> Diagnosis;
        
        Diagnosis [label="Final Assessment", shape=doubleoctagon, fillcolor="#C8E6C9"];
    }
''')