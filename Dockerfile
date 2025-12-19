FROM python:3.11-slim

WORKDIR /app

# basic deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir mlflow scikit-learn pandas joblib

# copy MLProject
COPY MLProject /app/MLProject

# default command: run training project
CMD ["mlflow", "run", "/app/MLProject", "--env-manager=local"]
