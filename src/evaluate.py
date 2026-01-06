"""
Model evaluation for bank churn prediction project.
"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from train import train_model


def evaluate_model(csv_path):
    """
    Evaluates the trained model using classification metrics.
    """

    model, X_test, y_test, feature_names = train_model(csv_path)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation Results ---")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC (Area Under the Curve): {auc_score:.4f}")

    return model, feature_names


if __name__ == "__main__":
    evaluate_model("Churn_Modelling.csv")

