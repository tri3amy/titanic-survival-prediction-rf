# Titanic Survival Prediction
**Predicting survival using Random Forest**

![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-83.80%25-brightgreen)
Python, Scikit-Learn, Pandas, Data Visualization (matplotlib, seaborn), Joblib, SHAP, Jupyter.

## Repository Structure

```text
titanic-survival-prediction-rf/
│
├── data/
│   └── train.csv                 # Raw dataset from Kaggle
│
├── models/
│   └── rf_best_model.pkl         # Saved Random Forest model
│
├── notebooks/
│   └── portofolio_titanic.ipynb  # EDA, Feature Engineering, Training, and SHAP Evaluation
│
├── predict.py                    # Standalone Python script for inference
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Project Overview:
**Problem Statement:** This project aims to predict the survival of passengers aboard the Titanic during the 1912 tragedy.
**Objective:** The primary goal of this analysis is to predict passenger survival based on implicit historical rules, such as "women and children first" and social status (nobility titles). Additionally, the model allows for custom inference to predict the survival probability of fictional characters or new data entries based on specific passenger attributes.
**Key Result:**
Achieved an accuracy of 83.80% using the Random Forest algorithm. GridSearchCV was implemented to perform hyperparameter tuning, specifically optimizing tree depth and node splits. The model currently yields 19 False Negative (actual survivors predicted as deceased), indicating an area for feature refinement to minimize the False Negative rate.

Source:
Kaggle Machine Learning Competition.
Features:
Sex, Title, PClass (passenger Class), Cabin (availability), Fare, Age, Family size (SibSp, Parch), Embarked (Q, S, C).
Target:
Survived (1) / Did Not Survive (0).

## Data Cleaning:
Imputed missing values in the 'Age' feature using the median. The median was selected over the mean to account for a non-normal distribution and the presence of outliers.
Exploratory Data Analysis (EDA):
Visualizations revealed that 'Sex' and 'Pclass' significantly impacted survival rates. The historical "women and children first" protocol is evident in the data, with females and children showing much higher survival probabilities. First-class passengers also demonstrated a higher likelihood of survival. The 'Age' distribution was right-skewed, with the majority of passengers aged between 20 and 30. Passenger 'Title' strongly correlated with survival, notably reflected by the high mortality rate among adult males (Title: 'Mr').
Feature Engineering:
Created a new variable, 'Family-size', by aggregating 'SibSp' (siblings/spouses) and 'Parch' (parents/children). This significantly improved model accuracy, as the size of a passenger's traveling party heavily influenced rescue logistics and survival chances.

## Algorithm:
Random Forest Classifier. Selected for its robustness on smaller datasets and its inherent resistance to overfitting. This ensemble algorithm yields strong predictive performance with minimal initial modification.
Hyperparameter Tuning:
Utilized GridSearchCV to systematically exhaust and identify the optimal combination of hyperparameters. The best parameters found were 'max-depth:10' and 'n_estimators:50'. A maximum depth of 10 allows the model to capture sufficient complexity while acting as a regularizer to prevent overfitting. Utilizing 50 decision trees (n_estimators) provided a stable and computationally efficient ensemble model for a dataset of this size (891 instances).
## Metric:
Model performance evaluation based on the Confusion Matrix on the test set (179 instances):
**Confusion matrix:**  
**True Negative** : 95 (Correctly predicted non-survivors)
**True Positive** : 55 (Correctly predicted survivors)
**False Positive** : 10 (Incorrectly predicted as survived)
**False Negative** : 19 (Incorrectly predicted as not survived)

Accuracy: 83.80% ((55+95)/179= 83.80%)
Precision (Measures the reliability of a positive prediction): 55/(55+10)= 84.62% 
Recall (Measures the model's ability to identify all actual survivors):55/(55+19)= 74.32% 

## Practical insight:  
Based on SHAP (SHapley Additive exPlanations) analysis, 'Sex' and 'Title' exhibit the highest global feature importance. Feature values representing females and passengers with non-'Mr' titles heavily drive the model toward a higher probability of survival.
## Future Work: 
Reducing the False Negative rate to make the model more accurate and reliable. Potential optimization strategies to decrease False Negatives include Threshold Tuning (lowering the classification threshold), Class Weighting, and Group Feature Engineering (exploring deeper latent patterns in the data that have not yet been discovered).

## Requirements:
scikit-learn, pandas, matplotlib, seaborn, shap, joblib, jupyter.
Local environment installation steps: 
1.	Clone repository:
```bash
git clone https://github.com/tri3amy/titanic-survival-prediction-rf.git
```
2.	Navigate to project folder:
```bash
cd titanic-survival-prediction-rf
```
3.	Create and activate virtual environment: 
```bash
conda create -n titanic_env python=3.11 -y
```
activation for windows:
```bash
conda activate titanic_env
```
4.	Install dependencies:
```bash
pip install -r requirements.txt
```
5.	Launch jupyter notebook:
```bash
jupyter notebook
```
For prediction:
Dictionary:
Pclass; 1 = 1st, 2 = 2nd, 3 = 3rd
Sex; male = 0, female = 1 
Title; 1: Mr, 2: Miss, 3: Mrs, 4: Master, 5: Rare
Cabin; 0: No, 1: Yes
Embarked; S, C, Q 

You can import the prediction function to test custom data:
Predicting Caledon Hockley's fate (1st Class, Male, Age 30, Ticket Fare 150.0, Has Cabin, Family Size 2, Title 'Mr', Embarked 'S').
Predicting Margaret Brown's fate (1st Class, Female, Age 44, Ticket Fare 55.0, Has Cabin, Family Size 1, Title 'Mrs', Embarked 'C').

*Disclaimer: This project was built as a learning portfolio; AI assistants and community resources were utilized to construct and optimize the code.

Author: Tri Puji Utami

LinkedIn: 
