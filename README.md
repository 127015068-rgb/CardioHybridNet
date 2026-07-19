# ❤️ Heart Disease Prediction using Hybrid Deep Neural Networks (CNN-LSTM)

Reproduction and extension of *"A Robust Heart Disease Prediction System Using Hybrid Deep Neural Networks"* (Al Reshan et al., **IEEE Access, 2023**, DOI: https://doi.org/10.1109/ACCESS.2023.3328909) — implementing and benchmarking four classical ML models and four deep learning architectures (ANN, CNN, LSTM, and Hybrid CNN-LSTM) for cardiovascular disease classification on two public clinical datasets.

---

## 📌 Overview

Cardiovascular disease is one of the leading causes of death worldwide. This project develops a complete heart disease prediction pipeline using both Machine Learning and Deep Learning techniques.

The implementation reproduces the methodology proposed in the IEEE Access paper while benchmarking the performance of traditional ML algorithms against a Hybrid CNN-LSTM architecture using two publicly available heart disease datasets.

---

# 🏗️ Proposed System Architecture

![Heart Disease Architecture](./assets/Heart_disease_architecture.png)

### Workflow

The proposed framework follows these stages:

- Data Collection from two public heart disease datasets
- Data Preprocessing
  - Missing value handling
  - Feature Scaling (StandardScaler)
  - Binary class conversion
  - Outlier removal
- Feature Selection using Extra Trees Classifier
- Train-Test Split (80:20)
- Machine Learning Model Training
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree (DT)
  - Random Forest (RF)
- Deep Learning Model Training
  - Artificial Neural Network (ANN)
  - Convolutional Neural Network (CNN)
  - Long Short-Term Memory (LSTM)
  - Hybrid CNN-LSTM
- Performance Evaluation
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - Specificity
  - F1-Score
  - Matthews Correlation Coefficient (MCC)
  - ROC-AUC

The proposed Hybrid CNN-LSTM combines convolutional feature extraction with sequential learning to improve the prediction of cardiovascular disease from clinical tabular data.

---

## 🗂️ Datasets

| Dataset | Source | Records | Features |
|----------|--------|---------|----------|
| Cleveland Heart Disease | UCI / Kaggle | 303 | 13 |
| Comprehensive Heart Disease Dataset | Cleveland + Hungary + Switzerland + Long Beach VA + Statlog | 1,190 | 11 |

Dataset files are available inside the **dataset/** directory.

---

## 🧠 Models Implemented

### Machine Learning

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

### Deep Learning

- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Hybrid CNN-LSTM (Proposed Model)

---

## 📁 Repository Structure

```text
CardioHybridNet/
│
├── assets/
│   └── Heart_disease_architecture.png
│
├── dataset/
│   ├── processed_cleveland.csv
│   └── heart_statlog_cleveland_hungary_final.csv
│
├── notebook/
│   ├── Dataset1_ML.ipynb
│   ├── Dataset2_ML.ipynb
│   ├── DS1_ANN.ipynb
│   ├── DS1_CNN.ipynb
│   ├── DS1_LSTM.ipynb
│   ├── DS1_CNNxLSTM.ipynb
│   ├── DS2_ANN.ipynb
│   ├── DS2_CNN.ipynb
│   ├── DS2_LSTM.ipynb
│   └── DS2_CNNxLSTM.ipynb
│
├── Documentation.pdf
├── README.md
└── A_Robust_Heart_Disease_Prediction_System_Using_Hybrid_Deep_Neural_Networks.pdf
```

---

## 📊 Performance Comparison

### Comprehensive Dataset (1,190 Samples)

| Model | Accuracy |
|--------|-----------|
| SVM | 88.24% |
| KNN | 88.24% |
| Decision Tree | 86.13% |
| Random Forest | **98.32%** |
| ANN | 96.70% |
| CNN | 97.03% |
| LSTM | 96.50% |
| **Hybrid CNN-LSTM** | **97.53%** |

---

### Cleveland Dataset (303 Samples)

| Model | Accuracy |
|--------|-----------|
| SVM | 86.67% |
| KNN | 88.33% |
| Decision Tree | 86.67% |
| Random Forest | 88.33% |
| ANN | 91.30% |
| CNN | 91.30% |
| LSTM | 94.60% |
| **Hybrid CNN-LSTM** | **97.50%** |

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

---

## 🛠️ Technology Stack

- Python
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🚀 Getting Started

```bash
git clone https://github.com/127015068-rgb/CardioHybridNet.git

cd CardioHybridNet

pip install -r requirements.txt

jupyter notebook
```

Open any notebook inside the **notebook/** directory.

---

## 📖 Reference

Al Reshan, M. S., Amin, S., Zeb, M. A., Sulaiman, A., Alshahrani, H., & Shaikh, A.

*A Robust Heart Disease Prediction System Using Hybrid Deep Neural Networks.*

IEEE Access, 2023.

https://doi.org/10.1109/ACCESS.2023.3328909

---

## 📄 License

This repository is intended for academic and educational purposes only. Please cite the original paper and the UCI Heart Disease dataset if you build upon this work.
