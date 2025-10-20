# app_streamlit.py
# -------------------------------------------------------------
# White Blood Cell Classifier - Streamlit UI
# Run with:  streamlit run app_streamlit.py
# Keep this file in: D:\new project live imple\trustworthy
# Model file: D:\new project live imple\trustworthy\best_wbc_model.keras
# -------------------------------------------------------------

import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import pandas as pd

# -----------------------------
# Config & constants
# -----------------------------
st.set_page_config(page_title="WBC Classifier", page_icon="🧫", layout="wide")

DEFAULT_IMG_SIZE = 128   # must match what your model expects

# ✅ IMPORTANT: use a raw string for Windows paths (r"...") or double backslashes
DEFAULT_MODEL_PATH = r"D:\new project live imple\trustworthy\best_wbc_model.keras"

DEFAULT_CLASS_NAMES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]

# -----------------------------
# Streamlit cache fallback (supports old/new versions)
# -----------------------------
try:
    cache_resource = st.cache_resource          # Streamlit >= 1.18
except AttributeError:
    def cache_resource(func):                   # Older Streamlit
        return st.cache(allow_output_mutation=True)(func)

# -----------------------------
# Helpers
# -----------------------------
def preprocess_pil(img: Image.Image, img_size: int) -> np.ndarray:
    """Convert PIL image -> model tensor [1, H, W, 3] normalized to [0,1]."""
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.exception(e)
        st.stop()

def predict(model, img_tensor: np.ndarray):
    """Returns probs [C,] and predicted index."""
    probs = model.predict(img_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx

def make_prob_df(classes, probs):
    probs = np.asarray(probs, dtype=np.float32)
    return pd.DataFrame({"Class": classes, "Probability": probs}).set_index("Class")

# ---- Grad-CAM (optional) ----
def find_last_conv_layer(model):
    # Find the last 2D conv layer for Grad-CAM
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def grad_cam(model, img_tensor, class_index=None, conv_layer_name=None):
    """
    Returns a heatmap (H, W) in [0,1] using Grad-CAM.
    Only works if the model has a Conv2D layer accessible.
    """
    if conv_layer_name is None:
        conv_layer_name = find_last_conv_layer(model)
    if conv_layer_name is None:
        return None

    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    conv_outputs = conv_outputs * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.35):
    """Resize heatmap to image size, colorize, and overlay."""
    import cv2  # make sure opencv-python is installed
    h, w = pil_img.size[1], pil_img.size[0]  # PIL uses (w,h)
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    overlayed = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
    return Image.fromarray(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    img_size = st.number_input("Image size", value=DEFAULT_IMG_SIZE, min_value=64, max_value=512, step=32)
    classes_text = st.text_area(
        "Class names (comma-separated, in your model's output order)",
        value=",".join(DEFAULT_CLASS_NAMES),
        help="Important: must match the order used during training."
    )
    class_names = [c.strip() for c in classes_text.split(",") if c.strip()]
    show_table = st.checkbox("Show probability table", True)
    show_bars = st.checkbox("Show probability bar chart", True)
    show_gradcam = st.checkbox("Show Grad-CAM (if supported by model)", False)
    st.caption("Tip: If predictions look mis-labeled, double-check the class order.")

# -----------------------------
# Header & uploader
# -----------------------------
st.title("🧫 White Blood Cell Classifier")
st.markdown("Upload a microscopy image of a white blood cell to classify it into one of the target categories.")

uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])

# -----------------------------
# Load model once
# -----------------------------
model = load_model(model_path)

# -----------------------------
# Inference flow
# -----------------------------
col_left, col_right = st.columns([1, 1])

if uploaded is not None:
    pil_img = Image.open(uploaded)

    with col_left:
        st.subheader("Input Image")
        st.image(pil_img, use_column_width=True)

    # Preprocess & predict
    x = preprocess_pil(pil_img, img_size)
    probs, pred_idx = predict(model, x)
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
    confidence = float(probs[pred_idx]) * 100.0

    with col_right:
        st.subheader("Prediction")
        st.metric("Predicted Class", pred_label, f"{confidence:.2f}%")

        df = make_prob_df(class_names, probs)
        if show_bars:
            st.bar_chart(df, height=220)
        if show_table:
            st.table(df.style.format({"Probability": "{:.3f}"}))

    # Optional Grad-CAM
    if show_gradcam:
        heatmap = grad_cam(model, x, class_index=pred_idx, conv_layer_name=None)
        if heatmap is None:
            st.info("Grad-CAM could not be generated (no accessible Conv2D layer).")
        else:
            overlayed = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.35)
            st.subheader("Grad-CAM (model attention)")
            st.image(overlayed, use_column_width=True)
else:
    st.info("Upload a WBC image to get a prediction.")

st.caption("Note: The model expects RGB images resized to the specified Image size and normalized to [0,1].")
