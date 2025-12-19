import streamlit as st

# Global Page Config
st.set_page_config(
    page_title="Dementia Research Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- MAIN LANDING PAGE ---
st.title("🧠 Multimodal Dementia Research Platform")
st.markdown("### Integrating Acoustic, Linguistic, Clinical, and Imaging Data")
st.markdown("This system aggregates data from three distinct diagnostic pipelines to assist in the early detection and classification of dementia.")

st.divider()

# --- THREE MODULES OVERVIEW ---
c1, c2, c3 = st.columns(3)

with c1:
    st.info("### 🎙️ Module 1: Audio")
    st.write("""
    **Focus:** Detecting biomarkers in speech.
    - **Input:** Raw Audio (.wav)
    - **Features:** Pitch, Jitter, Speech Rate.
    - **Models:** Whisper + Random Forest.
    - **Goal:** Early screening via voice.
    """)
    st.success("👈 Go to **'Audio Analysis'**")

with c2:
    st.warning("### 📋 Module 2: Clinical")
    st.write("""
    **Focus:** Risk calc. from medical history.
    - **Input:** Patient Data (.csv)
    - **Features:** Age, SES, CDR, MMSE.
    - **Models:** Logistic Regression + SHAP.
    - **Goal:** Clinical stratification.
    """)
    st.success("👈 Go to **'Clinical Records'**")

with c3:
    st.error("### 🧠 Module 3: MRI Vision")
    st.write("""
    **Focus:** Visual brain structure analysis.
    - **Input:** MRI Scans (JPG/PNG)
    - **Features:** Cortical Atrophy.
    - **Models:** CNN (MobileNet) + GANs.
    - **Goal:** Stage classification.
    """)
    st.success("👈 Go to **'MRI Analysis'**")

st.divider()

# --- UPGRADED SYSTEM ARCHITECTURE DIAGRAM ---
st.subheader("🛠️ System Architecture")
st.write("The diagram below illustrates the multimodal data flow and model integration.")

st.graphviz_chart('''
    digraph {
        rankdir=LR;
        node [fontname="Helvetica", shape=box, style="filled,rounded", color=white, fontcolor=black];
        edge [color="#666666", arrowsize=0.8];

        # --- User Node ---
        User [label="🧑‍⚕️ User / Clinician", shape=ellipse, fillcolor="#212121", fontcolor=white, penwidth=0];

        # --- Subgraphs for Organization ---
        
        subgraph cluster_audio {
            label = "Module 1: Audio Analysis";
            style=filled;
            color="#E3F2FD"; # Light Blue
            AudioInput [label="🎙️ Audio Signal", fillcolor="#BBDEFB"];
            AudioModel [label="⚙️ RF & NLP Model", fillcolor="#64B5F6"];
        }

        subgraph cluster_clinical {
            label = "Module 2: Clinical Records";
            style=filled;
            color="#FFF3E0"; # Light Orange
            ClinicalInput [label="📋 CSV Metadata", fillcolor="#FFE0B2"];
            ClinicalModel [label="⚙️ Regression/SHAP", fillcolor="#FFB74D"];
        }

        subgraph cluster_mri {
            label = "Module 3: MRI Vision";
            style=filled;
            color="#F3E5F5"; # Light Purple
            MRIInput [label="🧠 MRI Scan", fillcolor="#E1BEE7"];
            MRIModel [label="⚙️ CNN & GAN", fillcolor="#BA68C8"];
        }

        # --- Final Decision Node ---
        Decision [label="✅ Comprehensive\nDiagnosis Report", shape=doubleoctagon, fillcolor="#66BB6A", fontcolor=white, width=2.5];

        # --- Connections ---
        User -> AudioInput;
        User -> ClinicalInput;
        User -> MRIInput;

        AudioInput -> AudioModel;
        ClinicalInput -> ClinicalModel;
        MRIInput -> MRIModel;

        AudioModel -> Decision [label=" Vocal Markers"];
        ClinicalModel -> Decision [label=" Risk Score"];
        MRIModel -> Decision [label=" Visual Class"];
    }
''')

st.caption("Figure 1: Multimodal Architecture integrating Audio, Text, and Image processing pipelines.")
