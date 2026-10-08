import os
import io
import json
import base64
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, jsonify, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
# Vercel rejects request bodies over 4.5 MB; fail early with a friendly message.
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

# --- Configuration & Model Load ---
# The Keras model is exported to ONNX so the deployed app doesn't need TensorFlow.
# Regenerate it with scripts/export_onnx.py after retraining.
MODEL_PATH = os.path.join(BASE_DIR, "waste_classifier.onnx")
CLASS_NAMES = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

# Read-only on Vercel: nothing is ever written to disk.
TEST_FOLDER = os.path.join(BASE_DIR, 'static', 'manual_dataset')


# =============================================================================
# WASTE DISPOSAL INFORMATION DATABASE
# Each class has: bin color, hazard level, icon, warning,
#                 disposal steps, fun fact, and a YouTube link
# =============================================================================

WASTE_INFO = {
    "battery": {
        "bin_color":    "Red",
        "bin_hex":      "#e74c3c",
        "hazard_level": "HIGH",
        "hazard_badge": "danger",
        "icon":         "🔋",
        "title":        "Battery",
        "warning":      "NEVER throw in regular bins. Contains toxic lead, mercury, and acid that can contaminate groundwater.",
        "steps": [
            "Do NOT puncture, crush, or dismantle the battery.",
            "Place in a sealed plastic bag if damaged or leaking.",
            "Drop off at electronics stores, malls, or designated battery collection bins.",
            "Check your local government's hazardous waste schedule for free collection days.",
        ],
        "fun_fact":     "A single car battery contains enough lead to contaminate 25,000 liters of drinking water.",
        "youtube_url":  "https://www.youtube.com/results?search_query=how+to+dispose+battery+waste+properly",
        "youtube_label": "Watch: Proper Battery Disposal Guide",
    },
    "biological": {
        "bin_color":    "Green",
        "bin_hex":      "#27ae60",
        "hazard_level": "LOW",
        "hazard_badge": "safe",
        "icon":         "🌿",
        "title":        "Biological / Organic",
        "warning":      "Keep separate from dry recyclables. Wet waste contaminates paper and cardboard.",
        "steps": [
            "Drain excess liquid from food scraps before disposal.",
            "Place in a green compost bin or biodegradable bag.",
            "Can be composted at home — great for garden fertilizer.",
            "Avoid mixing with plastic or non-organic materials.",
        ],
        "fun_fact":     "Composting food waste can reduce household carbon emissions by up to 500kg CO₂ per year.",
        "youtube_url":  "https://www.youtube.com/results?search_query=how+to+compost+food+waste+at+home",
        "youtube_label": "Watch: Home Composting for Beginners",
    },
    "cardboard": {
        "bin_color":    "Blue",
        "bin_hex":      "#2980b9",
        "hazard_level": "LOW",
        "hazard_badge": "safe",
        "icon":         "📦",
        "title":        "Cardboard",
        "warning":      "Wet or greasy cardboard (like pizza boxes) cannot be recycled — bin it as general waste.",
        "steps": [
            "Break down and flatten all boxes to save space.",
            "Remove any tape, staples, or plastic inserts.",
            "Keep dry — wet cardboard cannot be recycled.",
            "Place in the blue recycling bin or take to a paper recycling center.",
        ],
        "fun_fact":     "Recycling one ton of cardboard saves 17 trees and 7,000 gallons of water.",
        "youtube_url":  "https://www.youtube.com/results?search_query=how+to+recycle+cardboard+properly",
        "youtube_label": "Watch: Cardboard Recycling Guide",
    },
    "clothes": {
        "bin_color":    "Yellow",
        "bin_hex":      "#f39c12",
        "hazard_level": "LOW",
        "hazard_badge": "safe",
        "icon":         "👕",
        "title":        "Clothes / Textile",
        "warning":      "Do not throw wearable clothes in the trash — donate or upcycle them first.",
        "steps": [
            "If still wearable: donate to Goodwill, Caritas, or local shelters.",
            "If damaged: bring to textile recycling banks or clothing drop-off bins.",
            "Cut into rags for household cleaning instead of disposal.",
            "Only bin as general waste if the item is heavily soiled and non-reusable.",
        ],
        "fun_fact":     "The fashion industry produces 92 million tons of textile waste per year — donating extends a garment's life by 2+ years.",
        "youtube_url":  "https://www.youtube.com/results?search_query=how+to+recycle+old+clothes+and+textiles",
        "youtube_label": "Watch: Textile & Clothing Recycling",
    },
    "glass": {
        "bin_color":    "Green",
        "bin_hex":      "#16a085",
        "hazard_level": "MEDIUM",
        "hazard_badge": "caution",
        "icon":         "🍶",
        "title":        "Glass",
        "warning":      "Handle broken glass carefully — wrap in newspaper before disposal to prevent injuries.",
        "steps": [
            "Rinse glass bottles and jars to remove food or liquid residue.",
            "Remove metal lids — recycle separately with metals.",
            "Do NOT mix with ceramics, Pyrex, or light bulbs (different melting points).",
            "Place in green glass recycling bin or bring to a bottle bank.",
        ],
        "fun_fact":     "Glass can be recycled indefinitely without losing quality or purity.",
        "youtube_url":  "https://www.youtube.com/results?search_query=how+to+recycle+glass+waste+properly",
        "youtube_label": "Watch: Glass Recycling Explained",
    },
    "metal": {
        "bin_color":    "Blue",
        "bin_hex":      "#7f8c8d",
        "hazard_level": "LOW",
        "hazard_badge": "safe",
        "icon":         "🥫",
        "title":        "Metal",
        "warning":      "Sharp metal edges can cause injury — handle carefully and do not crush aerosol cans.",
        "steps": [
            "Rinse cans to remove food residue.",
            "Do NOT crush aerosol cans — they may still contain pressurized gas.",
            "Remove paper labels if possible.",
            "Place in blue recycling bin or bring to a scrap metal facility.",
        ],
        "fun_fact":     "Recycling aluminum uses 95% less energy than producing it from raw materials.",
        "youtube_url":  "https://www.youtube.com/results?search_query=how+to+recycle+metal+cans+scrap+metal",
        "youtube_label": "Watch: Metal Recycling Process",
    },
    "paper": {
        "bin_color":    "Blue",
        "bin_hex":      "#2980b9",
        "hazard_level": "LOW",
        "hazard_badge": "safe",
        "icon":         "📄",
        "title":        "Paper",
        "warning":      "Shredded paper is difficult to recycle — bag it separately or compost it instead.",
        "steps": [
            "Keep paper dry — wet paper cannot be recycled.",
            "Remove plastic windows from envelopes.",
            "Do not recycle thermal paper (receipts) — they contain BPA chemicals.",
            "Bundle newspapers and magazines and place in blue recycling bin.",
        ],
        "fun_fact":     "Recycling one ton of paper saves 17 trees and reduces water usage by 50%.",
        "youtube_url":  "https://www.youtube.com/results?search_query=how+to+recycle+paper+waste+properly",
        "youtube_label": "Watch: Paper Recycling Guide",
    },
    "plastic": {
        "bin_color":    "Blue",
        "bin_hex":      "#2980b9",
        "hazard_level": "MEDIUM",
        "hazard_badge": "caution",
        "icon":         "🧴",
        "title":        "Plastic",
        "warning":      "Not all plastics are recyclable — check the recycling number (1–7) on the bottom.",
        "steps": [
            "Check the recycling number: #1 (PET) and #2 (HDPE) are most accepted.",
            "Rinse containers to remove food or liquid.",
            "Flatten bottles and containers to save bin space.",
            "Remove caps — some facilities require caps separated.",
            "Place in blue recycling bin or drop at plastic collection point.",
        ],
        "fun_fact":     "Less than 10% of all plastic ever produced has been recycled. Proper sorting makes a real difference.",
        "youtube_url":  "https://www.youtube.com/results?search_query=how+to+recycle+plastic+waste+properly",
        "youtube_label": "Watch: Plastic Recycling & Sorting",
    },
    "shoes": {
        "bin_color":    "Yellow",
        "bin_hex":      "#f39c12",
        "hazard_level": "LOW",
        "hazard_badge": "safe",
        "icon":         "👟",
        "title":        "Shoes / Footwear",
        "warning":      "Do not throw wearable shoes in the trash — donate or send to specialist recyclers.",
        "steps": [
            "If still wearable: donate to shelters, Caritas, or shoe banks.",
            "Check if brands like Nike or Adidas have take-back recycling programs.",
            "Worn-out rubber soles can be recycled into playground surfaces.",
            "Only bin as general waste if the shoe is completely beyond use.",
        ],
        "fun_fact":     "Nike's Reuse-A-Shoe program has recycled over 30 million pairs of shoes into sports surfaces.",
        "youtube_url":  "https://www.youtube.com/results?search_query=how+to+recycle+old+shoes+donate",
        "youtube_label": "Watch: Shoe Recycling & Donation",
    },
    "trash": {
        "bin_color":    "Black",
        "bin_hex":      "#2c3e50",
        "hazard_level": "MEDIUM",
        "hazard_badge": "caution",
        "icon":         "🗑️",
        "title":        "General Trash",
        "warning":      "General waste goes to landfill — always double-check if any component can be separated and recycled.",
        "steps": [
            "Before binning: check if any part is recyclable (plastic cap, glass bottle, paper wrapping).",
            "Separate any recyclable components before disposal.",
            "Use biodegradable bin liners where possible.",
            "Place in the black general waste bin for council collection.",
        ],
        "fun_fact":     "The average person generates 4.4 lbs of trash per day. Reducing by just 10% saves 160 lbs per year.",
        "youtube_url":  "https://www.youtube.com/results?search_query=reduce+general+waste+at+home+tips",
        "youtube_label": "Watch: How to Reduce General Waste",
    },
}




# =============================================================================
# INFERENCE HELPERS
# =============================================================================

def preprocess_image(source):
    """source: a file path or file-like object. Returns a (1, 224, 224, 3) float32 array in [0, 255]."""
    img = Image.open(source).convert("RGB")
    img = img.resize((224, 224))
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)

def predict_probs(img_arr):
    """Returns the 10 class probabilities for a preprocessed image."""
    return session.run(None, {INPUT_NAME: img_arr})[0][0]

def make_preview(raw_bytes, max_side=512):
    """Small base64 JPEG data URI so an upload can be shown without saving it."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/live')
def live_mode():
    # The browser captures the camera and sends frames to /predict, so the
    # disposal info is handed to the page once instead of per request.
    return render_template('live.html', waste_info=WASTE_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Classifies one JPEG/PNG frame posted as the raw request body (live mode)."""
    raw = request.get_data()
    if not raw:
        return jsonify(error="No image data received."), 400
    try:
        probs = predict_probs(preprocess_image(io.BytesIO(raw)))
    except (UnidentifiedImageError, OSError):
        return jsonify(error="Could not read that image."), 400
    return jsonify(classes=CLASS_NAMES, probs=[float(p) for p in probs])

@app.route('/evaluation')
def evaluation_report():
    json_path = os.path.join(app.static_folder, 'evaluation_stats.json')
    if not os.path.exists(json_path):
        stats = {
            "overall_acc": 0, "total_samples": 0,
            "class_labels": [], "class_accs": [], "top_mistakes": []
        }
    else:
        with open(json_path, 'r') as f:
            stats = json.load(f)
    return render_template('evaluation_report.html', stats=stats)

@app.errorhandler(413)
def too_large(_e):
    return render_template('manual.html', error="That image is too large (max 4 MB). Try a smaller photo."), 413

@app.route('/manual', methods=['GET', 'POST'])
def manual_mode():
    result = None
    image_url = None
    batch_results = None

    if request.method == 'POST':

        # --- Handle Batch Test ---
        if 'run_batch' in request.form:
            test_images = [
                "battery.jpg", "cardboard.jpg", "clothes.jpg",
                "glass.jpg", "metal.jpg", "paper.jpg",
                "plastic.jpg", "shoes.jpeg", "trash.jpg"
            ]
            batch_results = []
            for img_name in test_images:
                path = os.path.join(TEST_FOLDER, img_name)
                if os.path.exists(path):
                    preds = predict_probs(preprocess_image(path))
                    top_idx = np.argmax(preds)
                    cls = CLASS_NAMES[top_idx]
                    with open(path, "rb") as fh:
                        thumb = make_preview(fh.read(), max_side=200)
                    batch_results.append({
                        "filename":         img_name,
                        "top1_class":       cls,
                        "top1_confidence":  round(float(preds[top_idx]) * 100, 2),
                        "url":              thumb,
                        "is_uncertain":     float(preds[top_idx]) < 0.60,
                        "info":             WASTE_INFO.get(cls),
                    })
            return render_template('manual.html', batch_results=batch_results)

        # --- Handle User Upload (processed in memory, never saved) ---
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                raw = file.read()
                try:
                    preds = predict_probs(preprocess_image(io.BytesIO(raw)))
                    image_url = make_preview(raw)
                except (UnidentifiedImageError, OSError):
                    return render_template('manual.html', error="That file isn't a readable image."), 400

                top1_idx = np.argmax(preds)
                top3_indices = np.argsort(preds)[::-1][:3]
                cls = CLASS_NAMES[top1_idx]

                result = {
                    "top1_class":       cls,
                    "top1_confidence":  round(float(preds[top1_idx]) * 100, 2),
                    "top3":             [(CLASS_NAMES[i], round(float(preds[i]) * 100, 2)) for i in top3_indices],
                    "is_uncertain":     float(preds[top1_idx]) < 0.60,
                    "info":             WASTE_INFO.get(cls),
                }

    return render_template('manual.html', result=result, image_url=image_url)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
