import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Set page configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# Header and instructions
st.markdown(
    """
    <h1 style='color:#1976D2; text-align:center; font-weight: 700; margin-bottom: 0.1em;'> Brain Tumor Detection using YOLO</h1>
    <p style='text-align:center; color:#555; font-size:1.1em; margin-top: 0;'>Upload an MRI image below to detect and highlight potential brain tumors.</p>
    <hr style='border:1px solid #ddd; margin-bottom:2em;'/>
    """,
    unsafe_allow_html=True
)

# File uploader  
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Load and show original image
    image = Image.open(uploaded_file).convert('RGB')

    # Load YOLO model
    model = YOLO('best.pt')

    # Run model inference
    results = model(np.array(image))
    annotated_frame = results[0].plot()

    # Use two equal columns with gap between
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)
        st.caption("Original MRI you submitted.")

    with col2:
        st.image(annotated_frame, caption="Detection Result", use_container_width=True)
        st.caption("Detected tumor region highlighted.")
        
        # Aggregate max confidence per class
        class_confidences = defaultdict(float)
        for obj in results[0].boxes.data.tolist():
            cls_id = int(obj[5])
            conf = obj[4]
            if conf > class_confidences[cls_id]:
                class_confidences[cls_id] = conf

        # Show detection confidences grouped by class
        st.markdown(
            "<div style='background-color:#e0f7fa; border-radius:10px; padding:15px; margin-top:15px;'>"
            "<b>Detection Confidence (highest per class):</b>",
            unsafe_allow_html=True
        )
        if class_confidences:
            for cls_id, conf in class_confidences.items():
                st.markdown(
                    f"<div style='background-color:#fceabb; border-radius:5px; padding:6px; margin-top:7px;'>"
                    f"<b>{model.names[cls_id]}</b> &nbsp;&nbsp;"
                    f"<span style='color:#388e3c;'><b>Confidence:</b> {conf:.2%}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("No tumor detected in this image.")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr>
    <div style='text-align:center; color:#888; font-size:0.9em;'>© 2025 Brain Tumor Detection App – Powered by Ultralytics YOLO</div>
    """,
    unsafe_allow_html=True
)
