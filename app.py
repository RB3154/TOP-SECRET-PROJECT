import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import datetime
import tempfile
import numpy as np
import gradio as gr
import streamlit.components.v1 as components
from download_model import download_model

st.set_page_config(page_title="Weld Defect Detector - YOLOv8", layout="centered")
st.title("🔧 Weld Defect Detector - YOLOv8")
st.markdown("Select a detection mode below:")

# ⬇ Ensure model exists
download_model()
model = YOLO("best (1).pt")

# === Tabs ===
tab1, tab2 = st.tabs(["🖼️ Image Upload", "📷 Webcam Live Detection"])

# === TAB 1 ===
with tab1:
    st.header("📁 Upload Image for Prediction")
    uploaded_file = st.file_uploader("Upload JPG, PNG", type=["jpg", "jpeg", "png"])
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, step=0.01)

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_img.write(uploaded_file.read())
        temp_img_path = temp_img.name

        with st.spinner("🧠 Running inference..."):
            results = model.predict(source=temp_img_path, conf=conf_thresh, iou=0.45)

            for r in results:
                output_img = r.plot()
                st.image(Image.fromarray(output_img), caption="🔍 Detection Output", use_column_width=True)

                if r.boxes:
                    st.subheader("📝 Detected Classes")
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        st.write(f"✅ `{r.names[cls]}` with confidence `{conf:.2f}`")
                else:
                    st.warning("😕 No defects detected.")

        os.remove(temp_img_path)

# === TAB 2 ===
with tab2:
    st.header("📷 Live Detection from Webcam")

    def gradio_live_detect(frame):
        image = Image.fromarray(frame)
        result = model.predict(source=image, conf=0.25, iou=0.45)[0]
        return result.plot()

    # Launch Gradio inline
    with st.spinner("⚡ Loading webcam..."):
        gradio_html = gr.Interface(
            fn=gradio_live_detect,
            inputs=gr.Image(source="webcam", streaming=True),
            outputs="image",
            live=True
        ).launch(share=False, inline=True, inbrowser=False, prevent_thread_lock=True)

        components.html(gradio_html, height=640)
