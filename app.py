import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

st.set_page_config(page_title="PneumoniaScan AI", page_icon="🫁", layout="wide")

st.title("PneumoniaScan AI")
st.subheader("Rapid Chest X-Ray Screening for Pneumonia")
st.warning("⚠️ **Disclaimer:** This project is a prototype and is not medical advice.")

tab1, tab2, tab3 = st.tabs(["Live X-Ray Scanner", "The Problem & Research", "How It Works & Results"])

with tab1:
    st.header("Upload a Chest X-Ray")
    st.write("Designed for ER doctors needing a fast second opinion within critical 4-hour windows.")
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Scan", use_container_width=True)
            
        with col2:
            st.subheader("Screening Result")
            # NOTE: Plug your actual model prediction script/function here once exported from Colab!
            st.warning("⚠️ Model prediction logic needs to be connected here so it doesn't hardcode a fake score.")

with tab2:
    st.header("Why Early Detection Matters")
    st.markdown("""
    * **Radiologist Shortages:** Rising imaging volumes and workforce backlogs delay emergency care.
    * **Delayed Diagnosis (17% Rate):** Subtle early-stage pneumonia signs lead to delays in antibiotic timing (*PubMed Central*).
    * **High Uncertainty (31.8%):** Standard chest X-rays frequently come back inconclusive (*American Journal of Roentgenology*).
    """)

with tab3:
    st.header("Model Architecture & Results")
    st.write("**Dataset:** Chest X-ray dataset (5,856 images)")
    st.write("**Model:** ResNet-18")
    st.write("**Performance:** Baseline scored X% | Our Model scored Y% on unseen test data.")
