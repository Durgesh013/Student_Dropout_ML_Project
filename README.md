# 🎓 Student Dropout Prediction (Machine Learning Project)

## 📌 Project Overview

This project builds a machine learning model to predict whether a student is likely to drop out. Early identification of at-risk students helps institutions take preventive actions and improve retention.

---

## 📂 Project Structure

```
Student-Dropout-ml-project
├── data/
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
├── src/
│   └── student_dropout.py
├── notebook/
│   └── student_dropout.ipynb
├── models/
├── README.md
```

---

## ⚙️ Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🧠 ML Pipeline

### 1. Data Preprocessing

* Dropped irrelevant column: `Student_ID`
* Removed missing values using `dropna()`
* Split dataset into train/test (80/20)

### 2. Feature Engineering

* **Categorical Features** → OneHotEncoder
* **Numerical Features** → StandardScaler
* **Skewed Features** → PowerTransformer (Yeo-Johnson)

### 3. Pipeline

Used `ColumnTransformer` + `Pipeline` to ensure consistent preprocessing and modeling.

---

## 🤖 Model Used

* Logistic Regression
* Parameters:

  * `solver = "lbfgs"`
  * `max_iter = 1000`
  * `class_weight = "balanced"`
  * `random_state = 42`

---

## 📊 Model Performance

### Classification Report

```
Accuracy: 0.75

Class 0 (Non-Dropout):
Precision: 0.90
Recall: 0.75
F1-score: 0.82

Class 1 (Dropout):
Precision: 0.48
Recall: 0.74
F1-score: 0.58
```

---

## 📉 Confusion Matrix
<img src="https://raw.githubusercontent.com/Durgesh013/Student_Dropout_ML_Project/main/images/Confusion-Matrix.png" width="600">
---

## 📈 ROC Curve (AUC ≈ 0.82)

<img src="https://raw.githubusercontent.com/Durgesh013/Student_Dropout_ML_Project/main/images/ROC-Curve.png" width="600">

---

## 📈 Precision-Recall Curve

<img src="https://raw.githubusercontent.com/Durgesh013/Student_Dropout_ML_Project/main/images/Precision-Recall-Curve.png" width="600">

---

## 🔍 Key Insights

* Model achieves **75% accuracy**
* Strong **recall (0.74)** for dropout class → good at identifying at-risk students
* Lower precision (0.48) → more false positives

---

## 🚀 Future Improvements

* Improve precision using:

  * Advanced models (Random Forest, XGBoost)
  * Hyperparameter tuning
  * Better feature engineering
* Handle class imbalance with SMOTE / SMOTENC
* Deploy model using Streamlit or Flask

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone https://github.com/<your-username>/Student-Dropout-ml-project.git
cd Student-Dropout-ml-project
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the script

```
python src/student_dropout.py
```

---

## 📌 Notes

* Dataset is not included (add your own CSV file)
* Ensure file name: `student_dropout_dataset_v3.csv`

---

## 👨‍💻 Author

Durgesh Swarnakar
