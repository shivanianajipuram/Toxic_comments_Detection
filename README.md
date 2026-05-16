# Toxic Comment Detection Web App

A machine learning based toxic comment detection web application that identifies whether user-entered comments are Toxic or Non-Toxic using Natural Language Processing (NLP) techniques and an interactive web interface.

The application helps detect harmful, abusive, offensive, or inappropriate comments commonly found on online platforms and social media. It demonstrates the implementation of NLP preprocessing, text vectorization, machine learning model training, and real-time toxicity prediction through a user-friendly Streamlit dashboard.

---

# Dataset Used

The project uses the Jigsaw Toxic Comment Classification Dataset for training the toxicity detection model.

Dataset contains:
- User comments
- Toxicity labels (Toxic / Non-Toxic)

---

# NLP Techniques Used

- Text Cleaning
- Stopword Removal
- TF-IDF Vectorization
- Toxic Comment Classification
- Machine Learning Prediction

---

# Tech Stack

## Frontend
- Streamlit

## Backend / Machine Learning
- Python
- Scikit-learn
- NLP Preprocessing

## Libraries Used
- Pandas
- NLTK
- Joblib
- Matplotlib
- NumPy

## Machine Learning Model
- Logistic Regression

---

# Tools & Platforms

- VS Code
- Git
- GitHub
- Streamlit Cloud / Render

---

# Project Structure

```txt
ToxicCommentDetection/
│
├── app.py
├── train.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── runtime.txt
├── train.csv
└── README.md
```

---

# Features

- Detects toxic comments from user text input
- Classifies comments as Toxic or Non-Toxic
- Real-time toxicity prediction
- Toxicity confidence score visualization
- Interactive Streamlit dashboard
- NLP preprocessing pipeline
- TF-IDF based text vectorization
- User-friendly interface
- Machine learning model integration
- Offensive comment detection system

---

# How to Run Locally

## Clone the repository

```bash
git clone https://github.com/shivanianajipuram/Toxic_comments_Detection.git
```

---

## Open the project folder

```bash
cd Toxic_comments_Detection
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the application

```bash
streamlit run app.py
```

---

# Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy
- TF-IDF Vectorizer
- Logistic Regression
- Matplotlib

---

##Live demo
```bash
https://toxic-comments-detection-uc93.onrender.com/
```
