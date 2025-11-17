import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def analyze_document(image_path: str):
    """
    Placeholder analysis function.
    Replace with: load model, run inference, compute Grad-CAM & SHAP outputs.
    Returns:
      - tampering_score (float)
      - gradcam_path (str) (a image path to overlay)
      - shap_summary (dict)
    """
    # Simulated score
    tampering_score = round(random.uniform(0.0, 1.0), 3)

    # Create a fake Grad-CAM heatmap image for demo
    base = Image.open(image_path).convert("RGBA")
    w, h = base.size
    heatmap = Image.new("RGBA", base.size, (255,0,0,80))  # translucent red overlay
    # draw a fake hotspot rectangle (simulate tampered region)
    draw = ImageDraw.Draw(heatmap)
    rect = (int(w*0.2), int(h*0.1), int(w*0.65), int(h*0.35))
    draw.rectangle(rect, outline=(255,255,0,200), width=6)
    combined = Image.alpha_composite(base, heatmap)

    gradcam_path = os.path.join(UPLOAD_DIR, f"gradcam_{os.path.basename(image_path)}")
    combined.save(gradcam_path)

    # Fake SHAP summary
    shap_summary = {
        "text_anomaly": round(random.uniform(0, 0.5), 3),
        "logo_edit": round(random.uniform(0, 0.5), 3),
        "photo_morph": round(random.uniform(0, 0.5), 3),
    }

    return tampering_score, gradcam_path, shap_summary
