"""
DROWZY: Driver Drowsiness Detection with NLP-Based Fatigue Text Analysis
A Hybrid Artificial Intelligence System Integrating Computer Vision and Natural Language Processing for Driver Safety

Author: B. Sylesh (Roll No: 124cs0072)
Institution: Indian Institute of Information Technology Design & Manufacturing (IIITDM) Kurnool
Department: Department of Computer Science & Engineering
"""

import os
import sys
import json
import subprocess
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="DROWZY | CV + NLP Driver Safety System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1d3557;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #457b9d;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }
    .author-banner {
        background-color: #f1f5f9;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #1d3557;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
    }
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">🚗 DROWZY: Driver Drowsiness Detection with NLP</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Hybrid Artificial Intelligence System Integrating Computer Vision and Natural Language Processing for Driver Safety</div>', unsafe_allow_html=True)

st.markdown("""
<div class="author-banner">
    <strong>Academic Project:</strong> B.Tech in Computer Science & Engineering | <strong>Candidate:</strong> B. Sylesh (124cs0072)<br>
    <strong>Institution:</strong> Indian Institute of Information Technology Design & Manufacturing (IIITDM) Kurnool | <strong>Academic Year:</strong> 2026–2027
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select System Module:",
    [
        "Module 2: NLP Driver Fatigue Text Analyzer",
        "Module 1: Computer Vision Drowsiness Detection",
        "Model Evaluation & Performance",
        "System Architecture & Documentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status**:\n"
    "- Computer Vision: OpenCV + dlib 68 Landmarks\n"
    "- NLP Engine: TF-IDF + Logistic Regression\n"
    "- Safety Logic: Dual-Threshold Alarm & Guidance"
)

# -------------------------------------------------------------
# MODULE 2: NLP DRIVER FATIGUE TEXT ANALYZER
# -------------------------------------------------------------
if page == "Module 2: NLP Driver Fatigue Text Analyzer":
    st.header("🧠 Module 2: NLP-Based Driver Fatigue Text Analyzer")
    st.markdown(
        "Enter natural language driver statements or voice-to-text transcripts to assess the driver's cognitive "
        "and physical state across five operational categories: **FATIGUE**, **ALERT**, **REST_REQUEST**, **STRESS**, and **NEUTRAL**."
    )

    from nlp.fatigue_classifier import DriverFatigueClassifier
    classifier = DriverFatigueClassifier()

    # Quick test prompts
    st.subheader("Try a Sample Statement:")
    cols = st.columns(5)
    sample_texts = [
        "I am feeling very sleepy and exhausted.",
        "I feel alert, fresh, and ready to drive.",
        "Let us pull over at the next rest stop for coffee.",
        "Traffic is terrible and giving me severe anxiety.",
        "The cruise control is set to fifty five miles per hour."
    ]
    sample_labels = ["Fatigue Sample", "Alert Sample", "Rest Request", "Stress Sample", "Neutral Sample"]

    selected_sample = None
    for i, col in enumerate(cols):
        if col.button(sample_labels[i], key=f"sample_{i}"):
            selected_sample = sample_texts[i]

    # Input Box
    default_text = selected_sample if selected_sample else ""
    user_input = st.text_area(
        "Enter driver sentence or transcript:",
        value=default_text,
        placeholder="e.g., I cannot keep my eyes open and I keep yawning...",
        height=100
    )

    if st.button("🔍 Analyze Driver Text", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a sentence before analyzing.")
        else:
            with st.spinner("Processing natural language utterance..."):
                result = classifier.predict(user_input)

            cat = result["predicted_category"]
            conf = result["confidence_score"]
            color = result["category_color"]
            rec = result["recommendation"]
            probs = result["class_probabilities"]
            prep = result["preprocessing"]

            st.markdown("---")
            st.subheader("Analysis Results")

            rcol1, rcol2 = st.columns([1, 1.2])

            with rcol1:
                st.markdown(f"""
                <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 1.2rem; border-radius: 6px;">
                    <span style="font-size: 0.9rem; text-transform: uppercase; color: {color}; font-weight: bold;">Predicted Category</span>
                    <h2 style="color: {color}; margin: 0.2rem 0;">{cat}</h2>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>Confidence Score:</strong> {conf * 100:.1f}%</p>
                    <p style="margin: 0; font-size: 0.95rem; color: #1e293b;"><strong>Safety Recommendation:</strong><br>{rec}</p>
                </div>
                """, unsafe_allow_html=True)

                st.caption(f"ℹ️ {result['disclaimer']}")

            with rcol2:
                st.markdown("#### Class Probability Distribution")
                prob_df = pd.DataFrame({
                    "Category": list(probs.keys()),
                    "Probability (%)": [p * 100 for p in probs.values()]
                }).sort_values(by="Probability (%)", ascending=True)

                st.bar_chart(data=prob_df.set_index("Category"))

            # Step-by-Step Preprocessing Walkthrough
            st.markdown("---")
            with st.expander("🔬 View NLP Preprocessing & Pipeline Breakdown", expanded=True):
                st.markdown("This educational inspection shows how the input text is transformed step-by-step before machine learning inference:")
                
                step_cols = st.columns(4)
                step_cols[0].metric("1. Lowercased & Cleaned", prep["cleaned"][:25] + ("..." if len(prep["cleaned"]) > 25 else ""))
                step_cols[1].metric("2. Raw Tokens Extracted", len(prep["tokens"]))
                step_cols[2].metric("3. Stopwords Removed", len(prep["stopwords_removed"]))
                step_cols[3].metric("4. Lemmatized Tokens", len(prep["lemmatized_tokens"]))

                st.markdown(f"**Lemmatized Canonical Tokens:** `{prep['lemmatized_tokens']}`")
                st.markdown(f"**Final Processed Text fed to TF-IDF:** `{prep['final_processed_text']}`")
                st.caption("Feature Vectorizer: TF-IDF (Unigrams + Bigrams, Sublinear Scaling) | Classifier: Balanced Multinomial Logistic Regression")

# -------------------------------------------------------------
# MODULE 1: COMPUTER VISION DROWSINESS DETECTION
# -------------------------------------------------------------
elif page == "Module 1: Computer Vision Drowsiness Detection":
    st.header("👁️ Module 1: Computer Vision Driver Drowsiness Detection")
    st.markdown(
        "The computer vision subsystem continuously tracks the driver's face and eyes using a live webcam stream, "
        "localizes 68 facial landmarks, computes the **Eye Aspect Ratio (EAR)**, and triggers an audible and visual alarm "
        "when prolonged eye closure (microsleep) is detected."
    )

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("Operational Parameters & Algorithms")
        st.markdown("""
        - **Face Detector:** dlib HOG + Linear SVM Frontal Face Detector
        - **Facial Landmark Model:** `shape_predictor_68_face_landmarks.dat` (68 anatomical coordinates)
        - **Eye Coordinate Indices:**
          - Left Eye: Landmark indices 36 to 41 (6 landmarks)
          - Right Eye: Landmark indices 42 to 47 (6 landmarks)
        - **Eye Aspect Ratio (EAR) Formula:**
        """)
        st.latex(r"EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}")
        st.markdown("""
        - **EAR Threshold:** `0.30` (Eyes considered closed when EAR < 0.30)
        - **Consecutive Frame Threshold:** `48 frames` (~1.6 seconds at 30 FPS)
        - **Alarm Mechanism:** Visual contour drawing + Red warning text + `winsound.Beep(2500 Hz, 1000 ms)`
        """)

    with c2:
        st.subheader("Run Standalone Computer Vision System")
        st.info(
            "You can execute the live OpenCV webcam detection directly on your local machine using the command line "
            "or by clicking the button below."
        )

        st.code("python main.py", language="bash")

        if st.button("▶️ Launch OpenCV Webcam Detection (main.py)"):
            st.warning("Launching webcam stream window. Note: Press 'q' inside the OpenCV window to exit the camera feed.")
            try:
                subprocess.Popen([sys.executable, "main.py"])
                st.success("Webcam monitor launched in a separate window.")
            except Exception as e:
                st.error(f"Failed to launch webcam process: {e}")

# -------------------------------------------------------------
# MODEL EVALUATION & PERFORMANCE
# -------------------------------------------------------------
elif page == "Model Evaluation & Performance":
    st.header("📊 Model Evaluation & Experimental Metrics")
    st.markdown(
        "The NLP Driver Fatigue Classifier was trained on a balanced dataset of 300 labeled driver sentences "
        "across five classes and evaluated using an independent 20% stratified test split (60 samples). "
        "Below are the verified experimental metrics obtained directly from execution."
    )

    # Load metrics
    metrics_path = "nlp/evaluation_results.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy", f"{m['accuracy'] * 100:.2f}%")
        m2.metric("Macro Precision", f"{m['precision_macro'] * 100:.2f}%")
        m3.metric("Macro Recall", f"{m['recall_macro'] * 100:.2f}%")
        m4.metric("Macro F1-Score", f"{m['f1_macro'] * 100:.2f}%")

        st.markdown("---")
        st.subheader("Per-Class Experimental Performance")
        cr = m["classification_report"]
        perf_data = []
        for cat in m["classes"]:
            cdata = cr[cat]
            perf_data.append({
                "Category": cat,
                "Precision (%)": round(cdata["precision"] * 100, 2),
                "Recall (%)": round(cdata["recall"] * 100, 2),
                "F1-Score (%)": round(cdata["f1-score"] * 100, 2),
                "Test Support": int(cdata["support"])
            })
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

        st.markdown("---")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.subheader("Confusion Matrix")
            if os.path.exists("confusion_matrix.png"):
                st.image("confusion_matrix.png", use_container_width=True)
        with pcol2:
            st.subheader("Dataset Class Distribution")
            if os.path.exists("dataset_distribution.png"):
                st.image("dataset_distribution.png", use_container_width=True)
    else:
        st.error("Evaluation results file not found. Please run `python -m nlp.train_model` first.")

# -------------------------------------------------------------
# SYSTEM ARCHITECTURE & DOCUMENTATION
# -------------------------------------------------------------
elif page == "System Architecture & Documentation":
    st.header("🏗️ System Architecture & Workflow")
    st.markdown(
        "DROWZY integrates physical physiological monitoring (via Computer Vision) and driver sentiment/cognitive state analysis "
        "(via Natural Language Processing) to form a multi-modal Advanced Driver Assistance System (ADAS)."
    )

    if os.path.exists("system_architecture.png"):
        st.image("system_architecture.png", caption="Drowzy Hybrid Dual-Modality Architecture", use_container_width=True)

    st.markdown("---")
    st.subheader("NLP Pipeline Architecture")
    if os.path.exists("nlp_pipeline_diagram.png"):
        st.image("nlp_pipeline_diagram.png", caption="Natural Language Processing Pipeline Flow", use_container_width=True)

    st.markdown("---")
    st.subheader("Dual-Modality Complementarity")
    st.markdown("""
    | Operational Dimension | Pillar 1: Computer Vision | Pillar 2: Natural Language Processing |
    | :--- | :--- | :--- |
    | **Input Modality** | Video frames from webcam (RGB/Grayscale) | Text sentences / voice-to-text transcripts |
    | **Physiological Target** | Involuntary eye closure, blink duration, microsleeps | Cognitive fatigue, psychological stress, proactive rest intent |
    | **Detection Horizon** | Immediate / Reactive (seconds before accident) | Proactive / Anticipatory (minutes or hours before microsleep) |
    | **Key Metric** | Eye Aspect Ratio (EAR < 0.3 for 48 frames) | TF-IDF n-gram vector + Multinomial Logistic Regression probability |
    | **Intervention Action** | Loud emergency audio alarm (2500 Hz beep) | Contextual safety advice, rest stop recommendation, route pause |
    """)
