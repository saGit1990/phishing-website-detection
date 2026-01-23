import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.utils import print_header, ensure_artifacts_dir

def train_logistic(X_train, X_test, y_train, y_test, save_path):
    print_header("TRAINING XGBOOST (MLflow Enabled)")

    ensure_artifacts_dir(save_path)
    params = "Log Reg Basic"

    with mlflow.start_run(run_name="LogReg_Phishing"):

        model = LogisticRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        # 🔹 Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("f1_score", report["1"]["f1-score"])

        # 🔹 Save locally
        joblib.dump(model, save_path)

        # 🔹 Log model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="Logistic_model")

        print("Log Reg Accuracy:", acc)
        print("Classification Report:\n", classification_report(y_test, preds))
        print(f"XGBoost model saved → {save_path}")