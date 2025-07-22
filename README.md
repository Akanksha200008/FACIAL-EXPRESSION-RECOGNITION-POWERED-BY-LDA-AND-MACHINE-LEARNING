# Advance-Data-Mining
Kaggle dataset: https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition

# 😊 Facial Emotion Detection using LDA & SVM

This project is a lightweight, web-based application that detects **eight core human emotions** from grayscale facial images. It uses a machine learning pipeline with **Linear Discriminant Analysis (LDA)** and **Support Vector Machine (SVM)** to classify emotions, supported by a Python backend and a simple web interface.

---

## 🔍 What This Project Does

- Takes a grayscale image as input (via upload)
- Preprocesses and transforms it into a feature vector
- Applies LDA to reduce dimensionality
- Classifies the image using an SVM model
- Returns the predicted emotion from the following:
  - `Anger`, `Contempt`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`

---

## 💡 Why We Built It

Facial emotion recognition is key in human-computer interaction, mental health assessment, and behavioral analytics. This project demonstrates how simple yet effective classical ML techniques like LDA and SVM can be applied to this task in a lightweight, interpretable way — without the need for deep learning or large infrastructure.

---

## 🧰 Technologies Used

### 🔙 Backend
- **Python**
- **Flask** – Lightweight web framework
- **Flask-CORS** – For cross-origin API access
- **OpenCV** – Image preprocessing
- **NumPy / Pandas** – Data processing
- **Scikit-learn** – LDA, SVM, Evaluation metrics
- **Joblib** – Model serialization

### 🌐 Frontend
- **HTML5 & CSS3** – User interface
- **JavaScript** – API integration and UI feedback

---

## 📁 Folder Structure
emotion-detection-app/
- ├── frontapp.py # Flask API server for emotion prediction
- ├── initialcode.py # ML model training and evaluation script
- ├── index.html # Web interface for uploading images
- ├── lda_model.pkl # Trained LDA dimensionality reduction model
- ├── svm_model.pkl # Trained SVM emotion classifier
- ├── uploads/ # Folder to store uploaded test images
- ├── metrics_table.csv # Precision, recall, and F1-score per emotion
- ├── confusion_matrix.csv # Confusion matrix of model predictions


---

## 🚀 How to Run the Project

### 1.Clone the Repository
git clone https://github.com/Akanksha200008/FACIAL-EXPRESSION-RECOGNITION-POWERED-BY-LDA-AND-MACHINE-LEARNING

### 2.Create Virtual Environment
- python -m venv venv
- source venv/bin/activate  # Windows: venv\Scripts\activate

### 3.Install Dependencies
- pip install -r requirements.txt

- If not present, generate it with:
- pip freeze > requirements.txt

### 4. Optional(Train the Model)
python initialcode.py

This script:
- Loads facial images and emotion labels
- Applies preprocessing and LDA
- Trains an SVM classifier
- Saves the models and evaluation results

### 5. Start the Flask App
- python frontapp.py
- The API will start at http://127.0.0.1:5000/

### 6. Open the Web UI
Open index.html in a browser. Upload an image to get real-time emotion prediction from the model.

### 📊 Model Performance Summary
| Emotion   | Precision |  Recall  | F1-Score |
|----------|-----------|----------|-------|
|Happy|	0.94	|0.95|	0.94	|
|Neutral |	0.89 |	0.90	| 0.89	|
|Fear|	0.87	|0.84	| 0.85|

- Accuracy: ~93%
- F1 Macro Avg: ~92%
- Evaluation tools: classification_report, confusion_matrix from scikit-learn

### ✨ Key Highlights
- High accuracy using classical ML methods
- No need for GPUs or deep learning
- Portable and lightweight for demo or educational use
- Clean web interface for ease of testing and interaction

### 📈 Future Improvements
- Add face detection using OpenCV Haar cascades
- Extend dataset and retrain for robustness
- Integrate deep learning models like CNNs for comparison
- Deploy app to a public cloud or serverless function

---

