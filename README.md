
# Student Dropout Prediction

## Project Description
This project aims to develop a machine learning model to predict student dropout. By identifying students at risk of dropping out, educational institutions can implement targeted interventions to improve retention rates and support student success.

## Data Processing Steps

### Data Loading and Initial Cleaning
The dataset `student_dropout_dataset_v3.csv` was loaded into a pandas DataFrame using `pd.read_csv()`. The `Student_ID` column was dropped as it's not relevant for model training, and any rows with missing values were removed using `dropna()`.

### Data Splitting
The dataset was divided into features `X` (all columns except 'Dropout') and the target variable `y` ('Dropout'). These were then split into training and testing sets using `train_test_split` with an 80/20 ratio (`test_size=0.2`) and `random_state=42` for reproducibility.

### Feature Categorization
The features were categorized as follows:
*   **Categorical Columns**: `['Gender', 'Internet_Access', 'Part_Time_Job', 'Scholarship', 'Semester', 'Department', 'Parental_Education']`
*   **Numerical Columns**: `['Age', 'Study_Hours_per_Day', 'Attendance_Rate', 'Travel_Time_Minutes', 'Stress_Index', 'GPA', 'Semester_GPA', 'CGPA']`
*   **Skewed Columns**: `['Assignment_Delay_Days', 'Family_Income']`

### Preprocessing Pipelines
Different preprocessing steps were applied based on the column type:
*   **Skewed Features**: Transformed using `PowerTransformer(method="yeo-johnson")` to handle skewed distributions.
*   **Categorical Features**: Encoded using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` to convert them into a numerical format suitable for machine learning algorithms.
*   **Numerical Features**: Scaled using `StandardScaler()` to standardize their ranges.

These individual preprocessing steps were organized into pipelines and then combined into a single `ColumnTransformer` to ensure consistent and efficient data transformation across all feature types.

## Model Architecture
A Logistic Regression model was chosen as the base classifier for this project. It was initialized with the following parameters:
*   `solver="lbfgs"`
*   `max_iter=1000`
*   `class_weight="balanced"` (to address potential class imbalance)
*   `random_state=42`

The Logistic Regression model was integrated into a `Pipeline` along with the `ColumnTransformer` for preprocessing. This `Pipeline` streamlines the workflow, ensuring that the same preprocessing steps are applied to new data before making predictions.

## Evaluation Results

### Classification Report
```
              precision    recall  f1-score   support

           0       0.90      0.75      0.82      1369
           1       0.48      0.74      0.58       435

    accuracy                           0.75      1804
   macro avg       0.69      0.74      0.70      1804
weighted avg       0.80      0.75      0.76      1804
```

### Confusion Matrix
```
[[1025  344]
 [ 114  321]]
```
<img src="https://raw.githubusercontent.com/<Durgesh013>/Student-Dropout-ml-project/main/images/confusion_matrix.png" width="600">
<img src="images/confusion_matrix.png" width="600">
### ROC Curve
AUC = 0.8201573428041276
<img src="https://raw.githubusercontent.com/<Durgesh013>/Student-Dropout-ml-project/main/images/roc_curve.png" width="600">

### Precision-Recall Curve
<img src="https://raw.githubusercontent.com/<Durgesh013>/Student-Dropout-ml-project/main/images/precision_recall_curve.png" width="600">

## Summary:

### Data Analysis Key Findings
*   The project successfully created a machine learning model to predict student dropout, integrating a `LogisticRegression` model with a comprehensive preprocessing pipeline.
*   The preprocessing steps included dropping irrelevant IDs, handling missing values, splitting data (80% training, 20% testing), and categorizing features into numerical, categorical, and skewed types.
*   Specific transformers were applied: `PowerTransformer` for skewed features, `OneHotEncoder` for categorical features, and `StandardScaler` for numerical features, all orchestrated by a `ColumnTransformer`.
*   The `LogisticRegression` model was initialized with `class_weight="balanced"` to address potential class imbalance.
*   **Model Performance Evaluation:**
    *   Overall accuracy was 0.75.
    *   For the non-dropout class (0): Precision was 0.90, Recall was 0.75, and F1-score was 0.82.
    *   For the dropout class (1): Precision was 0.48, Recall was 0.74, and F1-score was 0.58.
    *   The Confusion Matrix showed 1025 True Negatives, 344 False Positives, 114 False Negatives, and 321 True Positives.
    *   The ROC AUC score was 0.82.

### Insights or Next Steps
*   The model demonstrates a strong ability to identify actual dropout cases (recall of 0.74 for class 1), which is crucial for early intervention. However, the relatively low precision (0.48 for class 1) indicates a significant number of students are falsely identified as at risk.
*   Future work should focus on improving the precision for the dropout class to minimize false alarms and optimize resource allocation for interventions. This could involve exploring more advanced classification algorithms, further feature engineering, or alternative strategies for handling class imbalance.
