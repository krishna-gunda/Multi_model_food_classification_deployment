# 🍛 Multi-Model Food Intelligence Platform

> **Predict Indian & global food from images using 3 deep learning models — Custom CNN, VGG16, and ResNet50 — with live nutritional data powered by Redis.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Render-46E3B7?style=for-the-badge)](https://multi-model-food-classification-v5we.onrender.com/)
[![Models](https://img.shields.io/badge/Models-CNN_·_VGG16_·_ResNet50-FF6B35?style=for-the-badge)]()
[![Classes](https://img.shields.io/badge/Classes-34_Food_Categories-gold?style=for-the-badge)]()
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python)](https://python.org)

---

## 🌐 Live Application

**→ [https://multi-model-food-classification-v5we.onrender.com/](https://multi-model-food-classification-v5we.onrender.com/)**

> Deployed on **Render** · Kept alive 24/7 with **UptimeRobot**

---

## 🧠 What This Project Does

This is a full-stack **Food Intelligence Platform** that:

1. **Accepts a food image** uploaded by the user
2. **Classifies it** using one of three deep learning models the user selects
3. **Returns the predicted food class** with a confidence score
4. **Fetches nutritional data** (calories, carbs, fats, proteins, fiber per 100g) from a **Redis database**
5. **Displays per-class model metrics** (precision, recall, confidence) on demand via a Metrics button

---

## 🗂️ Project Structure

```
multi-model-food-classification/
│
├── models_training/                          # Training scripts for all 3 models
│   ├── custom_cnn_training.py
│   ├── vgg16_training.py
│   └── resnet50_training.py
│
├── static/                                   # CSS, JS, and frontend assets
├── templates/                                # HTML templates (Jinja2)
│
├── app.py                                    # Flask app — routing, inference, Redis queries
├── main.py                                   # Entry point
├── download_models.py                        # Downloads trained weights at startup
├── redis_connection.py                       # Injects food_data.json into Redis
│
├── food_data.json                            # Nutritional values for all 34 food classes
├── custom_cnn_metrics.json                   # Per-class metrics for Custom CNN
├── model_evaluation_resnet_results.json      # Per-class metrics for ResNet50
├── model_evaluation_results_vgg16.json       # Per-class metrics for VGG16
│
├── model_links.txt                           # Google Drive links to trained model weights
├── requirements.txt                          # Python dependencies
├── runtime.txt                               # Python 3.11.9
└── Procfile                                  # Render deployment config
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle — Food Image Classification Dataset](https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset) |
| **Total Classes** | 34 Indian & global food categories |
| **Images per Class** | 300 (balanced — original dataset was imbalanced) |
| **Training Split** | 250 images per class |
| **Validation Split** | 50 images per class |
| **Total Training Images** | 8,500 |
| **Total Validation Images** | 1,700 |

### 🍽️ Food Classes (34)

```
Fried Rice · Idli · Jalebi · Kadai Paneer · Kulfi · Masala Dosa · Momos
Paani Puri · Pakode · Pav Bhaji · Pizza · Samosa · Sushi · Biryani · Pasta
... and 19 more Indian & international food categories
```

> **Why balanced?** The original Kaggle dataset has unequal class sizes. To prevent model bias toward majority classes, each class was capped at **300 images** — giving all 34 foods equal representation during training.

---

## 🤖 Models

### 1. 🧪 Custom CNN (Built from Scratch)

A convolutional neural network designed and trained entirely from scratch for this specific food dataset.

- Multiple Conv2D + MaxPooling + BatchNorm + Dropout layers
- Softmax output layer over 34 classes
- No pre-trained weights — learned entirely from the food images

### 2. 🔵 VGG16 (Transfer Learning — Fine-Tuned)

Pre-trained on ImageNet. Fine-tuned for food classification.

- **Frozen layers:** Early convolutional blocks (ImageNet weights locked)
- **Trainable layers:** Last few convolutional layers + custom dense classification head
- Only domain-specific layers were retrained on our food dataset

### 3. 🔴 ResNet50 (Transfer Learning — Fine-Tuned)

Pre-trained on ImageNet with residual skip connections.

- **Frozen layers:** Base ResNet50 backbone (ImageNet weights locked)
- **Trainable layers:** Top layers fine-tuned on our 34-class food dataset
- Residual connections prevent vanishing gradients during fine-tuning

### 📈 Model Metrics

Each model's **per-class precision, recall, and confidence** are stored as JSON and displayed live when the user clicks the **📊 Metrics** button.

| Model | Metrics File |
|---|---|
| Custom CNN | `custom_cnn_metrics.json` |
| VGG16 | `model_evaluation_results_vgg16.json` |
| ResNet50 | `model_evaluation_resnet_results.json` |

---

## 🗄️ Redis — Nutritional Database

Food nutritional data is stored in **Redis** as a fast in-memory key-value store.

**Pipeline:**
1. `food_data.json` — contains nutritional values for all 34 classes
2. `redis_connection.py` — reads JSON and injects each food's data into Redis at startup
3. `app.py` — after prediction, queries Redis with the predicted class name
4. Nutritional values are returned and rendered instantly in the UI

**Nutritional fields returned (per 100g serving):**

| Field | Unit |
|---|---|
| Calories | kcal |
| Carbohydrates | g |
| Fats | g |
| Proteins | g |
| Fiber | g |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask |
| **Deep Learning** | TensorFlow / Keras |
| **Pre-trained Models** | VGG16, ResNet50 (ImageNet weights via Keras Applications) |
| **Database** | Redis |
| **Frontend** | HTML, CSS, JavaScript (Jinja2 templates) |
| **Model Storage** | Google Drive (downloaded via `download_models.py`) |
| **Deployment** | Render |
| **Uptime Monitoring** | UptimeRobot |
| **Runtime** | Python 3.11.9 |

---

## 🚀 Running Locally

### Prerequisites

- Python 3.11+
- Redis server installed and running
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/gundakrishna3/multi-model-food-classification.git
cd multi-model-food-classification

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / Mac
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Redis server
redis-server

# 5. Inject food nutritional data into Redis
python redis_connection.py

# 6. Download trained model weights from Google Drive
python download_models.py

# 7. Launch the Flask app
python main.py
```

Visit: `http://localhost:5000`

---

## 🖥️ How to Use

1. **Open** the live app: [https://multi-model-food-classification-v5we.onrender.com/](https://multi-model-food-classification-v5we.onrender.com/)
2. **Upload** a food image (JPG or PNG)
3. **Select a model** — Custom CNN, VGG16, or ResNet50
4. Click **⚡ Predict**:
   - Predicted food name is shown
   - Confidence score is displayed (e.g., 97.02%)
   - Nutritional breakdown appears (calories, carbs, fats, proteins, fiber per 100g)
5. Click **📊 Metrics** to see per-class precision, recall, and confidence for the chosen model

---

## ☁️ Deployment

Deployed on **[Render](https://render.com)** as a web service.

| File | Role |
|---|---|
| `Procfile` | Tells Render how to start the Flask app |
| `runtime.txt` | Pins runtime to Python 3.11.9 |
| `download_models.py` | Fetches model weights from Google Drive on startup (files too large for GitHub) |
| UptimeRobot | Pings the app every 5 minutes to prevent Render free-tier sleep |

---

## 📁 Key Files Explained

| File | Purpose |
|---|---|
| `app.py` | Core Flask app — image upload handling, model inference, Redis lookup, routing |
| `main.py` | Application entry point |
| `download_models.py` | Downloads `.h5` / `.keras` model weights from Google Drive at startup |
| `redis_connection.py` | Parses `food_data.json` and populates Redis with nutritional data |
| `food_data.json` | Nutritional database for all 34 food classes |
| `model_links.txt` | Google Drive URLs for each trained model file |
| `requirements.txt` | Python package dependencies |
| `Procfile` | Render start command |
| `runtime.txt` | Python version specification for Render |

---

## 🤝 Author

**G. Krishna** — Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gundakri-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/g-krishna630534/))
[![GitHub](https://img.shields.io/badge/GitHub-gundakrishna3-181717?style=flat&logo=github)](https://github.com/krishna-gunda))
[![Email](https://img.shields.io/badge/Email-gundakrishna338%40gmail.com-EA4335?style=flat&logo=gmail)](mailto:gundakrishna338@gmail.com)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

*Built with 🔥 by G. Krishna — Deep Learning · Computer Vision · MLOps*

</div>
