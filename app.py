import os

# ─────────────────────────────────────────────────────
# Auto Download Models
# ─────────────────────────────────────────────────────
required_models = [
    "models/vgg16_food_classification_model.onnx",
    "models/food_classification_custom_Resnetmodel.onnx",
    "models/food_classification_custom_model.onnx"
]

if not all(os.path.exists(model) for model in required_models):
    import download_models

import json
import numpy as np
import redis
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import onnxruntime as ort

# ─────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")

# ─────────────────────────────────────────────────────
# Redis Connection
# ─────────────────────────────────────────────────────
try:
    r = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
    r.ping()
    print("[Redis] Connected Successfully")
except Exception as e:
    print(f"[Redis Error] {e}")
    r = None

# ─────────────────────────────────────────────────────
# Model Paths (ONNX)
# ─────────────────────────────────────────────────────
MODEL_PATHS = {
    "custom_cnn":  "models/food_classification_custom_model.onnx",
    "resnet":      "models/food_classification_custom_Resnetmodel.onnx",
    "vgg16":       "models/vgg16_food_classification_model.onnx"
}

# ─────────────────────────────────────────────────────
# Metrics Files
# ─────────────────────────────────────────────────────
METRICS_FILES = {
    "custom_cnn": "custom_cnn_metrics.json",
    "resnet":     "model_evaluation_resnet_results.json",
    "vgg16":      "model_evaluation_results_vgg16.json"
}

# ─────────────────────────────────────────────────────
# Load Class Indices
# ─────────────────────────────────────────────────────
CLASS_INDICES_PATH = "class_indices.json"

if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    CLASS_NAMES = list(class_indices.keys())
    print("[Classes] Loaded from class_indices.json")
else:
    print("[WARNING] class_indices.json not found")
    CLASS_NAMES = []

# ─────────────────────────────────────────────────────
# Default Classes
# ─────────────────────────────────────────────────────
DEFAULT_CLASSES = [
    "burger", "butter_naan", "chai", "chapati", "chole_bhature",
    "dal_makhani", "dhokla", "fried_rice", "idli", "jalebi",
    "kadai_paneer", "kulfi", "masala_dosa", "momos", "paani_puri",
    "pakode", "pav_bhaji", "pizza", "samosa", "sushi",
    "biryani", "pasta", "salad", "sandwich", "soup",
    "steak", "tacos", "waffles", "noodles", "pancakes",
    "cheesecake", "ice_cream", "poha", "upma"
]

# ─────────────────────────────────────────────────────
# Dataset Cache
# ─────────────────────────────────────────────────────
_food_dataset_cache = None

def get_food_dataset():
    global _food_dataset_cache
    if _food_dataset_cache is not None:
        return _food_dataset_cache
    if r:
        try:
            raw = r.get("food_dataset")
            if raw:
                _food_dataset_cache = json.loads(raw)
                print("[Redis] Dataset Loaded Successfully")
                return _food_dataset_cache
        except Exception as e:
            print(f"[Redis Dataset Error] {e}")
    print("[Fallback] Using Default Dataset")
    return {cls: {} for cls in DEFAULT_CLASSES}

# ─────────────────────────────────────────────────────
# Class Names
# ─────────────────────────────────────────────────────
def get_class_names():
    if len(CLASS_NAMES) > 0:
        return CLASS_NAMES
    dataset = get_food_dataset()
    names = [str(k).strip().lower().replace(" ", "_") for k in dataset.keys()]
    if len(names) == 0:
        return DEFAULT_CLASSES
    return sorted(names)

# ─────────────────────────────────────────────────────
# Nutrition Fetch
# ─────────────────────────────────────────────────────
def fetch_nutrition(class_name):
    dataset = get_food_dataset()
    normalized_class = class_name.strip().lower().replace("_", " ")
    normalized_dataset = {
        str(k).strip().lower().replace("_", " "): v
        for k, v in dataset.items()
    }
    entry = normalized_dataset.get(normalized_class)
    if not entry:
        print(f"[Nutrition] No nutrition data for: {normalized_class}")
        return {
            "calories": "N/A", "carbohydrates": "N/A",
            "fats": "N/A", "proteins": "N/A", "fiber": "N/A"
        }
    return {
        "calories":      str(entry.get("calories", "N/A")),
        "carbohydrates": str(entry.get("carbs", "N/A")),
        "fats":          str(entry.get("fat", "N/A")),
        "proteins":      str(entry.get("protein", "N/A")),
        "fiber":         str(entry.get("fiber", "N/A"))
    }

# ─────────────────────────────────────────────────────
# Model Cache (ONNX Runtime Sessions)
# ─────────────────────────────────────────────────────
_model_cache = {}

def get_model(model_key):
    if model_key not in _model_cache:
        path = MODEL_PATHS.get(model_key)
        if not path:
            raise ValueError(f"Invalid model key: {model_key}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        print(f"[Model] Loading {model_key} from {path}")
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _model_cache[model_key] = session
        print(f"[Model] {model_key} Loaded Successfully")
    return _model_cache[model_key]

# ─────────────────────────────────────────────────────
# Get Input Size from ONNX Session
# ─────────────────────────────────────────────────────
def get_model_input_size(session):
    input_shape = session.get_inputs()[0].shape  # e.g. [None, 224, 224, 3]
    height = int(input_shape[1])
    width  = int(input_shape[2])
    return (width, height)

# ─────────────────────────────────────────────────────
# VGG16 Preprocessing (mean subtraction, no /255)
# ─────────────────────────────────────────────────────
def vgg_preprocess(arr):
    # BGR mean subtraction (same as Keras vgg16 preprocess_input)
    arr = arr[..., ::-1]  # RGB -> BGR
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    arr -= mean
    return arr

# ─────────────────────────────────────────────────────
# ResNet Preprocessing
# ─────────────────────────────────────────────────────
def resnet_preprocess(arr):
    # Same as Keras resnet50 preprocess_input (caffe mode)
    arr = arr[..., ::-1]  # RGB -> BGR
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    arr -= mean
    return arr

# ─────────────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────────────
def preprocess_image(file_bytes, model_key, session):
    target_size = get_model_input_size(session)
    print(f"[INFO] {model_key} expects {target_size}")

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)

    if model_key == "resnet":
        arr = resnet_preprocess(arr)
    elif model_key == "vgg16":
        arr = vgg_preprocess(arr)
    else:
        arr = arr / 255.0

    return arr, target_size

# ─────────────────────────────────────────────────────
# Metrics Loader
# ─────────────────────────────────────────────────────
def load_metrics(model_key, class_name):
    path = METRICS_FILES.get(model_key)
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[Metrics] File not found: {path}")
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        report = data.get("classification_report", data)
        normalized_report = {
            str(k).strip().lower().replace(" ", "_"): v
            for k, v in report.items()
        }
        normalized_class = class_name.strip().lower().replace(" ", "_")
        print(f"[Metrics] Searching for: {normalized_class}")
        metrics = normalized_report.get(normalized_class)
        if not metrics:
            print("[Metrics] Class not found")
            return None
        return {
            "precision": round(float(metrics.get("precision", 0)), 4),
            "recall":    round(float(metrics.get("recall", 0)), 4),
            "f1_score":  round(float(metrics.get("f1-score", 0)), 4),
            "support":   int(metrics.get("support", 0))
        }
    except Exception as e:
        print(f"[Metrics Error] {e}")
        return None

# ─────────────────────────────────────────────────────
# Home Route
# ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", classes=get_class_names())

# ─────────────────────────────────────────────────────
# Predict Route
# ─────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    model_key = request.form.get("model", "custom_cnn")

    try:
        session = get_model(model_key)
        file_bytes = file.read()
        arr, target_size = preprocess_image(file_bytes, model_key, session)

        # ONNX inference
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: arr})[0][0]

        class_names = get_class_names()
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        print(f"[Prediction] {predicted_class}")

        nutrition = fetch_nutrition(predicted_class)

        # Preview Image
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize(target_size)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        preview_b64 = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            "class":      predicted_class,
            "confidence": round(confidence, 2),
            "nutrition":  nutrition,
            "model":      model_key,
            "preview":    preview_b64
        })

    except Exception as e:
        print(f"[Prediction Error] {e}")
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────────────
# Metrics Route
# ─────────────────────────────────────────────────────
@app.route("/metrics", methods=["POST"])
def metrics():
    data = request.get_json()
    model_key  = data.get("model", "custom_cnn")
    class_name = data.get("class_name", "")
    if not class_name:
        return jsonify({"error": "class_name missing"}), 400
    result = load_metrics(model_key, class_name)
    if result is None:
        return jsonify({"error": "Metrics Not Found"}), 404
    return jsonify(result)

# ─────────────────────────────────────────────────────
# Classes Route
# ─────────────────────────────────────────────────────
@app.route("/classes")
def classes():
    return jsonify(get_class_names())

# ─────────────────────────────────────────────────────
# Run Flask App
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
