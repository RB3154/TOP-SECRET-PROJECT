import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
from download_model import download_model

download_model()
model = YOLO("best (1).pt")

def predict_live(frame):
    frame_rgb = Image.fromarray(frame)
    results = model.predict(source=frame_rgb, conf=0.25)[0]
    return results.plot()

gr.Interface(
    fn=predict_live,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="image",
    live=True,
    title="🔍 Weld Defect Live Detection"
).launch()
