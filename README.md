# Diabetes Prediction Using Machine Learning

A comprehensive machine learning project for predicting diabetes using the Pima Indians Diabetes Database. This project implements and compares multiple classification algorithms with advanced feature engineering and preprocessing techniques.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Overview

This project addresses the binary classification problem of predicting diabetes onset in patients. The implementation explores various machine learning algorithms and employs sophisticated preprocessing techniques to achieve optimal model performance.

### Key Highlights
- **Multiple ML Models**: Logistic Regression, Random Forest, Support Vector Machine (SVM), and XGBoost
- **Advanced Feature Engineering**: Leave-One-Out Encoding for categorical variables
- **Data Transformation**: Power transformations (Box-Cox, logarithmic) to normalize distributions
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and AUC-ROC

## Dataset

**Source**: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/code)

The dataset contains medical diagnostic measurements for 768 female patients of Pima Indian heritage, aged 21 and above.

### Features
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1) - target variable

## Features

### Data Preprocessing
1. **Missing Value Handling**:
   - Simple mean imputation for Glucose, BloodPressure, and BMI
   - Iterative imputation using KNN regressor for SkinThickness and Insulin

2. **Feature Engineering**:
   - Age categorization: Young Adult (≤30), Adult (31-50), Old (>50)
   - Pregnancy categorization: Never, One Time, Many Times
   - Leave-One-Out Encoding for categorical variables

3. **Data Transformation**:
   - Box-Cox power transformation for DiabetesPedigreeFunction
   - Logarithmic transformation for SkinThickness and Insulin
   - Standard scaling for numerical features

## Models

Four classification algorithms were implemented and compared:

### 1. Logistic Regression
- Max iterations: 500
- Class weight: balanced
- Regularization: L2 (default)

### 2. Random Forest
- Number of estimators: 75
- Max depth: 3
- Class weight: balanced
- Random state: 1126

### 3. Support Vector Machine (SVM)
- Kernel: RBF (default)
- C parameter: 0.3
- Class weight: balanced
- Probability estimates: enabled

### 4. XGBoost
- Learning rate: 0.01
- Number of estimators: 100
- Max depth: 3
- Subsample: 0.3
- Scale positive weight: 1.8692

## Results

### Model Performance Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.747 | 0.612 | 0.759 | 0.678 | 0.831 |
| **Random Forest** | 0.792 | 0.683 | 0.759 | 0.719 | 0.831 |
| **SVM** | 0.766 | 0.632 | 0.796 | 0.705 | 0.832 |
| **XGBoost** | **0.799** | **0.702** | 0.741 | **0.721** | 0.830 |

**Best Performing Model**: XGBoost achieved the highest accuracy (79.9%) and precision (70.2%), making it the most reliable model for this classification task.

## Installation

### Requirements
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
category-encoders
xgboost
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Diabetes-Prediction.git
cd Diabetes-Prediction
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn category-encoders xgboost
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/code) and place it in the project directory.

## Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook pratice1.ipynb
```

The notebook includes:
- Exploratory Data Analysis (EDA) with visualizations
- Data preprocessing pipeline
- Model training and evaluation
- Performance comparison across all models
- ROC curves and confusion matrices

## Project Structure

```
Diabetes-Prediction/
│
├── pratice1.ipynb          # Main Jupyter notebook with complete implementation
├── README.md               # Project documentation
└── diabetes.csv            # Dataset (to be downloaded)
```

## Methodology

1. **Data Exploration**: Statistical analysis and visualization to understand feature distributions
2. **Data Cleaning**: Handle zero values representing missing data
3. **Train-Test Split**: 80-20 split with random state for reproducibility
4. **Preprocessing**: Multi-stage imputation, transformation, and scaling
5. **Feature Engineering**: Create categorical features and apply target encoding
6. **Model Training**: Train multiple models with class imbalance handling
7. **Evaluation**: Compare models using multiple performance metrics

## Key Insights

- Missing values in medical measurements (represented as zeros) required careful imputation
- Class imbalance (65% negative, 35% positive) necessitated balanced class weights
- Feature transformations significantly improved model performance by normalizing distributions
- XGBoost with carefully tuned hyperparameters achieved the best overall performance
- All models achieved AUC-ROC scores above 0.83, indicating good discriminative ability

## Future Improvements

- Implement cross-validation for more robust performance estimation
- Explore additional ensemble methods (e.g., stacking, voting classifiers)
- Conduct hyperparameter tuning using GridSearchCV or Bayesian optimization
- Feature selection to identify most important predictors
- Collect additional features for improved prediction accuracy

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Dataset: UCI Machine Learning Repository
- Platform: Kaggle
- Libraries: scikit-learn, XGBoost, category-encoders
