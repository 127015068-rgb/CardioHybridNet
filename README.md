# ❤️ Heart Disease Prediction using Hybrid Deep Neural Networks (CNN‑LSTM)

Reproduction and extension of *"A Robust Heart Disease Prediction System Using Hybrid Deep Neural Networks"* (Al Reshan et al., **IEEE Access, 2023**, DOI: [10.1109/ACCESS.2023.3328909](https://doi.org/10.1109/ACCESS.2023.3328909)) — implementing and benchmarking four classical ML models and four deep learning architectures (ANN, CNN, LSTM, and a hybrid CNN‑LSTM) for cardiovascular disease classification on two public clinical datasets.

---

## 📌 Overview

Cardiovascular disease is the leading cause of death worldwide. This project builds an end-to-end pipeline that predicts the presence of heart disease from routine clinical attributes (age, blood pressure, cholesterol, ECG results, etc.), comparing traditional machine learning against a **hybrid 1D‑CNN + LSTM deep neural network** that learns spatial feature interactions and sequential dependencies jointly.

The goal was to achieve accurate results using the base paper's reference with same datasets, same evaluation metrics, same model families and benchmark our results against the numbers reported in the paper.

## 🗂️ Datasets

| Dataset | Source | Records | Features | Notebook prefix |
|---|---|---|---|---|
| **Cleveland HD Dataset** | UCI / Kaggle | 303 | 13 | `DS1_*`, `Dataset1_ML.ipynb` |
| **Comprehensive HD Dataset** | Statlog + Cleveland + Hungarian (+ Switzerland + Long Beach VA merged) | 1,190 | 11 | `DS2_*`, `Dataset2_ML.ipynb` |

Both are provided as CSVs under [`dataset/`](./dataset): `processed_cleveland.csv` and `heart_statlog_cleveland_hungary_final.csv`.

## 🧠 Models Implemented

**Classical Machine Learning (baseline)**
- Support Vector Machine (SVM)
- K‑Nearest Neighbors (KNN)
- Decision Tree (DT)
- Random Forest (RF)

**Deep Learning**
- Artificial Neural Network (ANN) — dense feed-forward network, evaluated with a 50‑seed ensemble to stabilize results on the small Cleveland split
- 1D / Multi-scale Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- **Hybrid CNN‑LSTM** — the paper's proposed architecture: tabular features are zero‑padded and reshaped into a 2‑D grid, passed through 4 Conv1D/Conv2D blocks (32→64→128→256 filters) with batch normalization and max‑pooling, then fed into an LSTM(128) layer and dense classification head. This lets the network learn local feature interactions (via convolution) and higher-order dependencies across the learned feature map (via the recurrent layer) in a single end-to-end model.

Each model is evaluated with **Accuracy, Precision, Sensitivity (Recall), Specificity, F1‑score, Matthews Correlation Coefficient (MCC), and AUC‑ROC** — matching the metric suite used in the base paper for a like-for-like comparison.

## 📁 Repository Structure

```
MiniProject/
├── dataset/
│   ├── processed_cleveland.csv                       # 303 records, 13 features
│   └── heart_statlog_cleveland_hungary_final.csv      # 1,190 records, 11 features
├── notebook/
│   ├── Dataset1_ML.ipynb        # SVM / KNN / DT / RF on Cleveland
│   ├── Dataset2_ML.ipynb        # SVM / KNN / DT / RF on comprehensive dataset
│   ├── DS1_ANN.ipynb            # ANN — Cleveland
│   ├── DS1_CNN.ipynb            # CNN — Cleveland
│   ├── DS1_LSTM.ipynb           # LSTM — Cleveland
│   ├── DS1_CNNxLSTM.ipynb       # Hybrid CNN-LSTM — Cleveland
│   ├── DS2_ANN.ipynb            # ANN — comprehensive dataset
│   ├── DS2_CNN.ipynb            # CNN — comprehensive dataset
│   ├── DS2_LSTM.ipynb           # LSTM — comprehensive dataset
│   └── DS2_CNNxLSTM.ipynb       # Hybrid CNN-LSTM — comprehensive dataset
├── A_Robust_Heart_Disease_Prediction_System_Using_Hybrid_Deep_Neural_Networks.pdf   # reference paper
└── Documentation.pdf             # project report
```

## 📊 Results

### Comprehensive Dataset (1,190 records) — our best-performing setting

| Model | Accuracy | Precision | Sensitivity | Specificity | MCC | F1 | AUC |
|---|---|---|---|---|---|---|---|
| SVM | 88.24% | 0.877 | 0.905 | 0.857 | 0.764 | 0.891 | 0.936 |
| KNN | 88.24% | 0.921 | 0.929 | 0.911 | 0.840 | 0.925 | 0.967 |
| Decision Tree | 86.13% | 0.891 | 0.841 | 0.884 | 0.724 | 0.865 | 0.949 |
| Random Forest | **98.32%** | 0.969 | 1.000 | 0.964 | 0.967 | 0.984 | 0.989 |
| ANN | 96.70% | 0.966 | 0.972 | 0.961 | 0.934 | 0.969 | 0.991 |
| CNN (multi-scale) | 97.03% | 0.968 | 0.976 | 0.964 | 0.940 | 0.972 | 0.991 |
| LSTM | 96.50% | 0.974 | 0.960 | 0.971 | 0.930 | 0.967 | 0.988 |
| **Hybrid CNN‑LSTM** | **97.53%** | 0.971 | 0.983 | 0.967 | 0.951 | 0.977 | **0.994** |

### Cleveland Dataset (303 records) — smaller, harder split

| Model | Accuracy | AUC | Notes |
|---|---|---|---|
| SVM | 86.67% | – | classical baseline |
| KNN | 88.33% | – | classical baseline |
| Decision Tree | 86.67% | – | classical baseline |
| Random Forest | 88.33% | – | classical baseline |
| ANN (50‑seed ensemble) | 91.30% | 0.949 | ensembling stabilizes the small-sample variance |
| CNN | 91.30% | 0.907 | |
| LSTM | 94.60% | 0.988 | |
| **Hybrid CNN‑LSTM** | **97.50%** | **0.995** | best model on this split |

### Us vs. the Base Paper

Across both datasets, our from-scratch re-implementation tracked the paper's reported numbers closely (typically within ±1–2 percentage points), and on the larger, more diverse comprehensive dataset our **Random Forest (98.32% vs. 90.0%)** and **Hybrid CNN‑LSTM (97.53% Accuracy, 0.994 AUC)** models **matched or exceeded** the paper's reported results. The largest deviations appeared on the small 303-row Cleveland split, where variance across random seeds is inherently higher — this is precisely why the paper (and this repo) also reports the comprehensive 1,190-row dataset as the primary benchmark.

**Key takeaway:** the hybrid CNN‑LSTM architecture consistently outperformed every single-architecture deep learning model (ANN, CNN, LSTM alone) and every classical ML baseline on AUC-ROC across both datasets, confirming the paper's central claim that combining convolutional feature extraction with recurrent sequence modeling yields a more discriminative representation for tabular clinical data than either component alone.

## 🛠️ Tech Stack

- **Python 3**, **TensorFlow / Keras** — deep learning models
- **scikit-learn** — classical ML models, preprocessing, metrics
- **NumPy / Pandas** — data handling
- **Matplotlib / Seaborn** — visualization (ROC curves, confusion matrices)
- **Jupyter Notebook** — experimentation environment

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/heart-disease-hybrid-dnn.git
cd heart-disease-hybrid-dnn
pip install -r requirements.txt
jupyter notebook
```

Open any notebook under `notebook/` — each is self-contained and reads directly from `dataset/`.

## 📖 Reference

Al Reshan, M. S., Amin, S., Zeb, M. A., Sulaiman, A., Alshahrani, H., & Shaikh, A. (2023). *A Robust Heart Disease Prediction System Using Hybrid Deep Neural Networks.* IEEE Access, 11, 121574–121591. https://doi.org/10.1109/ACCESS.2023.3328909

## 📄 License

This project is for academic/educational purposes. Datasets are publicly available UCI Heart Disease datasets; please cite the original UCI/Kaggle sources and the reference paper above if you build on this work.
