# Bank Customer Churn Prediction

Machine learning project to predict customer churn in a retail banking context, using supervised classification models and a structured end-to-end ML pipeline.

---

## Problem Statement

Customer churn represents a significant challenge for banks, as retaining existing customers is often more cost-effective than acquiring new ones.  
The goal of this project is to predict whether a customer is likely to leave the bank, enabling proactive retention strategies.

---

## Dataset

The dataset contains customer-level information, including:

- Demographic data
- Account balance
- Product usage
- Activity status

Target variable:
- **Exited**  
  - `1` → Customer left the bank  
  - `0` → Customer stayed  

---

## Approach

The project follows a structured machine learning workflow:

1. **Data preprocessing**
   - Removal of non-informative identifiers
   - One-hot encoding of categorical variables
   - Feature scaling

2. **Train-test split**
   - Applied before scaling to avoid data leakage

3. **Model training and evaluation**
   - Multiple classification models were trained and compared
   - Emphasis on metrics relevant to churn prediction

---

## Models and Results

The following models were trained and evaluated:

- Logistic Regression (baseline)
- Random Forest
- XGBoost

While Random Forest achieved the highest AUC score, **XGBoost provided the best recall for the churn class**, which is particularly important in churn prediction scenarios where failing to identify a customer at risk is more costly than a false positive.

### Performance Summary

- **Best AUC**: Random Forest (0.865)
- **Best Recall for churn class**: XGBoost (0.55)
- **Best F1-score for churn class**: XGBoost

This trade-off makes XGBoost the most suitable model from a business perspective.

---

## Key Takeaways

- Accuracy alone is not sufficient for churn prediction.
- Recall for the churn class is a more relevant metric in this context.
- Ensemble methods significantly outperform baseline linear models.
- XGBoost provides the best balance between predictive performance and business impact.

---

## Project Structure

```
.
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── Churn_Modelling.csv
├── requirements.txt
└── README.md
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python src/train.py
```

Evaluate model performance:

```bash
python src/evaluate.py
```

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib



