import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import datetime
import tempfile
from download_model import download_model

st.set_page_config(page_title="Weld Defect Detector - YOLOv8", layout="centered")
st.title("🔧 Weld Defect Detector - YOLOv8")
st.markdown("Select a detection mode below:")

# ⬇ Ensure model exists
download_model()
model = YOLO("best (1).pt")

# === Tabs ===
tab1, tab2 = st.tabs(["🖼️ Image Upload", "📸 Take Photo (Phone/Laptop)"])

# === TAB 1: Standard Image Upload ===
with tab1:
    st.header("📁 Upload Image for Prediction")
    uploaded_file = st.file_uploader("Upload JPG, PNG", type=["jpg", "jpeg", "png"])
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, step=0.01)

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(uploaded_file.read())
        img_path = temp_file.name

        with st.spinner("🧠 Running YOLOv8 inference..."):
            results = model.predict(source=img_path, conf=conf_thresh, iou=0.45)

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
        os.remove(img_path)

# === TAB 2: Take a Photo (Phone/Laptop Capture) ===
with tab2:
    st.header("📸 Capture Photo Using Device Camera")
    st.markdown("Most smartphones/laptops will show a **camera option** when you tap the uploader below.")

    camera_file = st.file_uploader("Take or upload a photo", type=["jpg", "jpeg", "png"], key="camera")
    if camera_file:
        st.image(camera_file, caption="Captured Photo", use_column_width=True)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(camera_file.read())
        img_path = temp_file.name

        with st.spinner("🧠 Running YOLOv8 inference..."):
            results = model.predict(source=img_path, conf=0.25, iou=0.45)
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
        os.remove(img_path)
