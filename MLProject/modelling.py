import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dagshub import dagshub_logger

# Tracking ke DagsHub
dagshub_logger.init("my-first-repo", "ferrikrisdiantoro", mlflow=True)
mlflow.set_experiment("Obesity Classification CI")

# Load data
data = pd.read_csv("obesity-classification_preprocessing.csv")
X = data.drop("Label_encoded", axis=1)
y = data["Label_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
