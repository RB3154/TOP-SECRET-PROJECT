import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
from download_model import download_model

st.set_page_config(page_title="Weld Defect Detector - YOLOv8", layout="centered")
st.title("🔧 Weld Defect Detector - YOLOv8")
st.markdown("Upload an image or capture a photo using your device's camera.")

# ⬇ Ensure model exists
download_model()
model = YOLO("best (1).pt")

# === Single Tab: Upload or Take Photo ===
st.header("📁 Upload Image/📸 Capture Photo Using Device Camera for Prediction")
uploaded_file = st.file_uploader("Choose or Take a Photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, step=0.01)

if uploaded_file:
    st.image(uploaded_file, caption="Selected Image", use_column_width=True)
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
