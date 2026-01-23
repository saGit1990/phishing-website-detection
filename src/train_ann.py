import joblib
import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from src.utils import  print_header

def train_ann(x_train, x_test, y_train, y_test, params, save_path):
    print_header("TRAINING ANN MLPClassifier with MLFLOw")

    # FIX: config compatibility
    params = params.copy()

    if 'hidden_layer' in params:
        params['hidden_layer_sizes'] = tuple(params['hidden_layers'])
        del params['hidden_layers']

    with mlflow.start_run(run_name="ANN_MLP_Phishing"):
        # log hyperparameters
        mlflow.log_params(params)

        ann = MLPClassifier(**params)
        ann.fit(x_train, y_train)

        pred = ann.predict(x_test)

        acc = accuracy_score(y_train,pred)
        report = classification_report(y_train,pred, output_dict=True)

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        mlflow.log_metric("f1_score", report['1']['f1-score'])

        # Save model
        joblib.dump(ann, save_path)

        # Log model to mlflow
        mlflow.sklearn.log_model(ann, artifact_path='ann_model')

        print("ANN Accuracy:", acc)
        print("Report:\n", classification_report(y_test, pred))
        print(f"ANN model saved → {save_path}")

