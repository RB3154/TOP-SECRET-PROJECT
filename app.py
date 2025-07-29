import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import shutil
import datetime
import csv
from download_model import download_model

st.set_page_config(page_title="Weld Defect Detector", layout="centered")

# ⬇ Download model if not present
download_model()

# 📦 Load model
model = YOLO("best (1).pt")
st.title("🛠️ Weld Defect Detector - YOLOv8")
st.markdown("Upload an image to detect defects or switch to webcam tab.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img_path = f"temp_{datetime.datetime.now().timestamp()}.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    results = model.predict(source=img_path, conf=conf_thresh, iou=0.45)

    for r in results:
        output_img = r.plot()
        st.image(Image.fromarray(output_img), caption="Predicted", use_column_width=True)

        st.write("### Detected Classes:")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"✅ {r.names[cls]} ({conf:.2f})")

    os.remove(img_path)
