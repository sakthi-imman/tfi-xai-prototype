# analyze.py (with GRAD-CAM + SHAP)
import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import cv2
import shap
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet50_tamper.h5")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
BG_DIR = os.path.join(BASE_DIR, "backgrounds")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)

print("Loading tampering model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------------------------------------
# GRAD-CAM MODEL
# ------------------------------------------------------------
LAST_CONV_LAYER = "conv5_block3_out"
grad_model = Model(
    inputs=model.inputs,
    outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
)

# ------------------------------------------------------------
# PREPROCESS FUNCTION
# ------------------------------------------------------------
def preprocess_pil_image(pil_img):
    pil_img = pil_img.resize(IMG_SIZE)
    x = image.img_to_array(pil_img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x

# ------------------------------------------------------------
# CREATE SHAP BACKGROUND DATASET
# ------------------------------------------------------------
def load_background_images(n=5):
    bg_images = []
    files = os.listdir(BG_DIR)
    if len(files) == 0:
        raise ValueError("ERROR: No images found in python-ai/backgrounds/")

    for f in files[:n]:
        img_path = os.path.join(BG_DIR, f)
        try:
            pil = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            arr = image.img_to_array(pil)
            bg_images.append(arr)
        except:
            pass

    bg_images = np.array(bg_images)
    bg_images = tf.keras.applications.resnet50.preprocess_input(bg_images)
    return bg_images

background = load_background_images()

# SHAP explainer (one-time load)
print("Initializing SHAP GradientExplainer...")
explainer = shap.GradientExplainer(model, background)

# ------------------------------------------------------------
# GRAD-CAM IMPLEMENTATION
# ------------------------------------------------------------
def grad_cam(pil_img, img_array, out_name=None):
    if out_name is None:
        out_name = f"gradcam_{uuid.uuid4().hex}.png"

    conv_outputs, predictions = grad_model(img_array)
    pred = predictions[0][0]
    class_idx = 1 if pred >= 0.5 else 0

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heat = np.mean(conv_outputs, axis=-1)
    heat = np.maximum(heat, 0)
    heat /= (heat.max() + 1e-9)

    img = np.array(pil_img)
    h, w, _ = img.shape

    heat = cv2.resize(heat, (w, h))
    heat = np.uint8(255 * heat)
    heatmap_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return out_path

# ------------------------------------------------------------
# SHAP EXPLANATION
# ------------------------------------------------------------
def shap_explain(pil_img, img_array, out_name=None):
    if out_name is None:
        out_name = f"shap_{uuid.uuid4().hex}.png"

    shap_values = explainer.shap_values(img_array)

    # SHAP returns a list for binary classification: [class0, class1]
    shap_img = shap_values[0][0]  # explanation for tampered/genuine

    img = np.array(pil_img).astype(np.float32)
    img = cv2.resize(img, IMG_SIZE)

    # Normalize SHAP values for visualization
    shap_norm = shap_img - shap_img.min()
    shap_norm /= (shap_norm.max() + 1e-9)
    shap_norm = cv2.resize(shap_norm, (img.shape[1], img.shape[0]))

    heat = np.uint8(255 * shap_norm)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img.astype(np.uint8), 0.6, heat_color, 0.4, 0)

    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, overlay)

    return out_path

# ------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ------------------------------------------------------------
def analyze_document(filepath):
    pil_img = Image.open(filepath).convert("RGB")
    img_array = preprocess_pil_image(pil_img)

    pred = model.predict(img_array)[0][0]
    tampering_score = float(pred)
    label = "tampered" if tampering_score >= 0.5 else "genuine"

    gradcam_path = grad_cam(pil_img, img_array)
    shap_path = shap_explain(pil_img, img_array)

    return {
        "tampering_score": tampering_score,
        "label": label,
        "gradcam": os.path.basename(gradcam_path),
        "shap": os.path.basename(shap_path)
    }
