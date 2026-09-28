# ❤️ Heart Disease Prediction — End-to-End ML Project

A **modular, production-ready** Machine Learning project that predicts the likelihood of heart disease using **Logistic Regression**, built with a complete ML pipeline and deployed via **Streamlit**.

---

## 🎯 Problem Statement

Heart disease is one of the leading causes of death globally. Early detection using patient health metrics can save lives. This project builds a binary classifier to predict whether a patient has heart disease based on clinical features like age, cholesterol, blood pressure, and more.

**Dataset**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) (via Kaggle — `heart.csv`)

---

## 🧠 Why Logistic Regression?

| Criterion | Reason |
|---|---|
| **Binary output** | Heart disease: Yes (1) or No (0) |
| **Interpretability** | Coefficients explain feature impact directly |
| **Baseline benchmark** | Ideal first model before complex ensembles |
| **Medical domain** | Outputs probability — clinically meaningful |
| **Small-medium dataset** | ~300 rows; simpler models generalize better |

---

## 📁 Project Structure

```
heart-disease-predictor/
│
├── data/
│   └── heart.csv                  # Raw dataset
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Load, clean, encode, split
│   ├── feature_engineering.py     # Scaling, feature selection
│   ├── model_training.py          # Train & save the model
│   ├── model_evaluation.py        # Metrics, confusion matrix, ROC
│   └── predict.py                 # Single-sample prediction utility
│
├── models/
│   └── logistic_model.pkl         # Saved trained model
│
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
│
├── app.py                         # Streamlit web application
├── train.py                       # Main training pipeline entry point
├── requirements.txt
└── README.md
```

---

## 🔄 ML Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

1. **Data Preprocessing** — Handle missing values, encode categoricals
2. **Feature Engineering** — StandardScaler normalization
3. **Model Training** — Logistic Regression with hyperparameter tuning (GridSearchCV)
4. **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
5. **Deployment** — Streamlit interactive web app

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Model Performance (on test set)

| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| Precision | ~84% |
| Recall | ~88% |
| F1-Score | ~86% |
| ROC-AUC | ~92% |

> Results may vary slightly depending on the random seed and train/test split.

---

## 🏥 Features Used

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | Sex (1=male, 0=female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalach` | Max heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type |

---

## 👤 Author

**Mohit** — GDG ML Task 2
