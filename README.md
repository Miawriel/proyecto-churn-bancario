# Bank Customer Churn Prediction

## Overview
This project focuses on predicting customer churn in the banking sector using machine learning techniques.  
Customer churn prediction is a real-world problem widely used by financial institutions to improve customer retention strategies.

The goal of this project is to identify customers who are more likely to leave the bank based on their demographic and behavioral data.

## Problem Statement
Customer acquisition is expensive. Losing existing customers directly impacts revenue.  
By predicting churn in advance, banks can take proactive actions such as targeted offers or personalized support.

This project frames churn prediction as a **binary classification problem**.

## Dataset
The dataset contains customer information such as:
- Credit score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of products
- Credit card status
- Estimated salary

Target variable:
- `Exited` (1 = customer left the bank, 0 = customer stayed)

## Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib / Seaborn
- XGBoost

## Project Workflow
1. Data cleaning and preprocessing  
2. Exploratory data analysis  
3. Feature encoding and scaling  
4. Model training and comparison  
5. Model evaluation using classification metrics  

## Results
Multiple models were tested.  
The final model prioritized **recall for the churn class**, aiming to correctly identify customers at risk of leaving.

## Future Improvements
- Hyperparameter optimization
- Model explainability (SHAP)
- Deployment as a simple web application

## Author
Mariel Lopez A.









