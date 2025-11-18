# app.py – updated for SHAP + GradCAM
import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from analyze import analyze_document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # save uploaded file
        ext = os.path.splitext(f.filename)[1].lower()
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        f.save(filepath)

        # run full AI + XAI analysis
        result = analyze_document(filepath)

        # build public URLs
        gradcam_url = f"/outputs/{result['gradcam']}"
        shap_url = f"/outputs/{result['shap']}"

        return jsonify({
            "tampering_score": result["tampering_score"],
            "label": result["label"],
            "gradcam_url": gradcam_url,
            "shap_url": shap_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
