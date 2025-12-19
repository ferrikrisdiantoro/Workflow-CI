import os
import argparse
import json
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

def load_processed(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="titanic_preprocessing")
    ap.add_argument("--experiment_name", default="titanic-modelling-basic")
    ap.add_argument("--run_name", default="baseline-run")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=1000)
    args = ap.parse_args()

    X_train, X_test, y_train, y_test = load_processed(args.data_dir)

    # =========================
    # DETEKSI MODE EKSEKUSI
    # =========================
    # Kalau dipanggil via `mlflow run`, MLFLOW_RUN_ID biasanya diset oleh MLflow Projects
    running_from_mlflow_project = bool(os.environ.get("MLFLOW_RUN_ID"))

    # =========================
    # CONFIG MLFLOW
    # =========================
    # Autolog tetap boleh
    mlflow.sklearn.autolog(log_models=True)

    # Kalau manual run (python modelling.py), baru set experiment
    if not running_from_mlflow_project:
        mlflow.set_experiment(args.experiment_name)

    # =========================
    # RUN HANDLING
    # =========================
    run_ctx = None
    if not running_from_mlflow_project:
        # manual mode → buat run baru
        run_ctx = mlflow.start_run(run_name=args.run_name)
    # project mode → jangan start_run, karena run sudah dibuat oleh mlflow run

    try:
        model = LogisticRegression(C=args.C, max_iter=args.max_iter)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # manual log metrics (biar eksplisit)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("roc_auc", float(auc))

        print("ACC:", acc)
        print("F1 :", f1)
        print("AUC:", auc)
        print("\nClassification report:\n", classification_report(y_test, y_pred))
        print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

        # artefak outputs/
        os.makedirs("outputs", exist_ok=True)

        metrics_path = os.path.join("outputs", "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": acc, "f1": f1, "roc_auc": auc}, f, indent=2)
        mlflow.log_artifact(metrics_path)

        rep_path = os.path.join("outputs", "classification_report.txt")
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(rep_path)

        cm_path = os.path.join("outputs", "confusion_matrix.txt")
        with open(cm_path, "w", encoding="utf-8") as f:
            f.write(str(confusion_matrix(y_test, y_pred)))
        mlflow.log_artifact(cm_path)

    finally:
        if run_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
