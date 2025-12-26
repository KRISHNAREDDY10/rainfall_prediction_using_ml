# 🌧️ Rainfall Prediction Using Machine Learning

## 📌 Project Overview
This project predicts **whether it will rain tomorrow (`RainTomorrow`)** using historical Australian weather data.  
It implements a **complete end-to-end machine learning pipeline**, covering data preprocessing, imbalance handling, feature engineering, multiple model training, evaluation, and comparison.

Special care was taken to ensure the project is **portable, reproducible, and macOS-compatible**, avoiding system-level dependencies that can cause runtime failures.

---

## 🎯 Problem Statement
Rainfall prediction is a **binary classification problem** with:
- High class imbalance (more “No Rain” days than “Rain” days)
- Significant missing values
- Mixed numerical and categorical features

The goal is to build reliable models that generalize well despite these challenges.

---

## 📦 Installation & Execution

### 1️⃣ Create Virtual Environment
```
MacOS:
python -m venv .venv
source .venv/bin/activate

Windows:
python -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure
```
Rainfall Prediction/
│
├── rainfall_prediction.ipynb
├── weatherAUS.csv
├── requirements.txt
├── README.md
└── .venv/
```

---

## 🛠️ Tech Stack
- **Language:** Python 3.11
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - CatBoost

---

## 🗂️ Dataset
- **Dataset:** Australian Weather Dataset (`weatherAUS.csv`)
- **Target Variable:** `RainTomorrow`
  - `0` → No Rain
  - `1` → Rain
- **Features Include:**
  - Temperature (Min/Max/9am/3pm)
  - Humidity
  - Pressure
  - Wind speed & direction
  - Cloud cover
  - Sunshine
  - Rainfall

---

## 🔄 Machine Learning Pipeline

### 1️⃣ Data Preprocessing
- Converted `Date` into `year`, `month`, and `day`
- Encoded target variables (`RainToday`, `RainTomorrow`) into binary form
- Mode imputation for categorical features
- Label Encoding for categorical variables

---

### 2️⃣ Handling Class Imbalance
- Applied **Random Oversampling** to balance the target classes
- Visualized class distribution before and after balancing

---

### 3️⃣ Missing Value Treatment
- Used **MICE (Multiple Imputation by Chained Equations)** via `IterativeImputer`
- Preserves statistical relationships between features
- More robust than mean/median imputation

---

### 4️⃣ Outlier Detection & Removal
- Used **Interquartile Range (IQR)** method
- Removed extreme values to improve model stability

---

### 5️⃣ Feature Scaling & Selection
- **MinMaxScaler** for Chi-Square feature selection
- **StandardScaler** for model training
- Selected top predictive features using **Chi-Square Test**

---

## 🧠 Models Implemented (OpenMP-Free)

The following models were chosen for **maximum portability and reliability on macOS**:

- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network (MLPClassifier)
- CatBoost Classifier

---

## ⚠️ Why XGBoost and LightGBM Were Excluded
XGBoost and LightGBM depend on **OpenMP (`libomp`)**, which is **not available by default on macOS** without additional system-level tooling (Homebrew/Conda).

To ensure:
- Clean execution on any macOS system
- No OS-level dependency errors
- Reproducibility for reviewers and recruiters

these models were intentionally excluded.

> Equivalent performance was achieved using **Random Forest and CatBoost**, making them practical and production-safe alternatives.

---

## 📊 Model Evaluation Metrics
Each model was evaluated using:

- **Accuracy**
- **ROC-AUC Score**
- **Cohen’s Kappa**
- **Classification Report**
- **ROC Curves**

These metrics provide a balanced evaluation, especially for imbalanced classification problems.

---

## 📈 Visualizations Generated
All visual outputs are automatically saved:

- Class imbalance (before & after oversampling)
- Missing value heatmap
- Correlation heatmap
- ROC curves for each model
- Model comparison charts (Accuracy & ROC-AUC)

---

## 🧪 Key Learnings
- Real-world data requires advanced imputation strategies
- Handling class imbalance is critical for weather prediction
- ROC-AUC and Cohen’s Kappa are more informative than accuracy alone
- Model portability and reproducibility matter in real environments

---

## 🚀 Future Improvements
- Hyperparameter tuning (GridSearch / Optuna)
- Cross-validation for robustness
- SHAP-based model explainability
- Deployment using Flask or FastAPI

---

## 📜 License
This project is intended for **educational and portfolio purposes**.
