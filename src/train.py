"""
Model training for bank churn prediction project.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from preprocess import preprocess_data


def train_models(csv_path):
    """
    Trains multiple models and returns them for comparison.
    """

    X_train, X_test, y_train, y_test, feature_names = preprocess_data(csv_path)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=42),
        "xgboost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, X_test, y_test, feature_names


if __name__ == "__main__":
    models, _, _, _ = train_models("Churn_Modelling.csv")
    print("All models trained successfully.")
