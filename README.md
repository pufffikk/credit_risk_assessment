# Credit Risk Assessment with Machine Learning

This project is focused on evaluating the creditworthiness of loan applicants using machine learning models. The dataset used is the **German Credit Dataset**. This repository contains the full analysis pipeline, including data preprocessing, exploratory analysis, model training, feature importance evaluation, and fairness considerations.

---

## 📌 Problem Statement

The goal is to develop a reliable classifier to determine whether a loan applicant is likely to be a credit risk (`1`) or not (`0`), based on various demographic, financial, and loan-related features.

---

## 📂 Dataset

The dataset is sourced from UCI's German Credit Data. It contains 1000 entries with 20 attributes, including categorical and numerical fields. We use the version where categorical variables have been one-hot encoded and numerical fields scaled.

---

## 🧹 Data Preparation

Key preprocessing steps:
- One-hot encoding of categorical features
- Standardization of numerical features
- Feature selection via correlation, variance, and permutation importance
- Removal of near-constant or irrelevant variables

---

## 🔍 Exploratory Data Analysis

- Correlation matrix to identify linear dependencies
- Distribution plots and boxplots for feature skew/outliers
- Class imbalance analysis and segment profiling

---

## 🤖 Model Development

We trained several classification models including:
- Logistic Regression (threshold tuning + class weighting)
- XGBoost
- Random Forest
- SVM and KNN (as baselines)
- SMOTE + Logistic Regression for imbalance handling

Each model was evaluated using metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC AUC
- Confusion Matrix

---

## 📊 Feature Importance

Was used `permutation_importance` from sklearn to assess how much each feature contributed to the predictive performance of the XGBoost model.

---

## 🧠 Ethical Considerations

- Manual review to avoid gender, race, or income bias
- Threshold tuning focused on balancing false negatives (high-risk loans classified as safe)
- Recommended SHAP for future explainability enhancements

---

## ✅ Conclusions

- XGBoost had the best ROC AUC (~0.82)
- Logistic Regression with threshold tuning had the best recall (0.81)
- Top features included: Credit Amount, Duration, Status, Employment, and Savings
- Final recommendation: use XGBoost with interpretability tools in production

---

## 📁 Files Included

- `Credit_Risk_Assessment.ipynb`: Main notebook with full pipeline
- `README.md`: This file
- `german.data`: Original dataset

---

## 🚀 How to Run

1. Clone the repository
2. Launch Jupyter Notebook
3. Run all cells in `Credit_Risk_Assessment.ipynb`

---

## 🧩 Future Work

- Use larger and more recent datasets
- Introduce real-time behavioral data
- Add model interpretability and fairness dashboards

---

## 🔗 Acknowledgements

- Dataset from UCI Machine Learning Repository
- Scikit-learn, XGBoost, and Seaborn used for modeling and visualization
