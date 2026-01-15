# Logistic Regression Plan for Churn Prediction

This plan outlines the steps to build a logistic regression model to predict customer churn.

## 1. Understand the Goal
The main objective is to build a classification model using logistic regression to predict whether a customer will churn based on the provided features. We will use the `churn-bigml-80.csv` file for training and `churn-bigml-20.csv` for testing.

## 2. Project Setup
- Use the existing `model.ipynb` file for the implementation.
- Ensure necessary libraries are installed: `pandas`, `scikit-learn`, `joblib`.

## 3. Data Loading and Initial Exploration
- Load the `churn-bigml-80.csv` (training data) and `churn-bigml-20.csv` (testing data) into pandas DataFrames.
- Examine the data:
    - Display the first few rows (`.head()`).
    - Check for missing values (`.isnull().sum()`).
    - Get summary statistics (`.describe()`).
    - Check data types (`.info()`).
    - Analyze the distribution of the target variable 'Churn'.

## 4. Data Preprocessing
- Identify categorical and numerical features.
- Create a preprocessing pipeline using `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`.
- The pipeline will handle:
    - **Categorical features**: One-hot encoding.
    - **Numerical features**: Scaling (e.g., using `StandardScaler`).

## 5. Model Training
- Define the logistic regression model (`sklearn.linear_model.LogisticRegression`).
- Create the full pipeline by combining the preprocessing pipeline and the logistic regression model.
- Train the model on the training data (`.fit()`).

## 6. Model Evaluation
- Use the trained model to make predictions on the test data (`.predict()`).
- Evaluate the model's performance using:
    - Accuracy score.
    - Classification report (precision, recall, f1-score).
    - Confusion matrix.

## 7. Model Persistence
- Save the trained pipeline to a file using `joblib` for future use.

## 8. Refinement (Optional)
- Based on evaluation results, consider feature engineering, trying different models, or hyperparameter tuning.
