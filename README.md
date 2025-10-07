# Breast Cancer Classifier

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Build](https://img.shields.io/github/actions/workflow/status/ROHANSRIVATSA/breastCancerPrediction/docker-build.yml?branch=main)](https://github.com/ROHANSRIVATSA/breastCancerPrediction/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GCP Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-orange.svg)](https://console.cloud.google.com/run)

**A fully functional ML and MLOps pipeline for Breast Cancer prediction, deployed as a FastAPI API with Docker and Cloud Run integration.**

Project Overview

This project demonstrates the end-to-end workflow for building, deploying, and scaling a machine learning model for Breast Cancer classification. It integrates data preprocessing, ML model training, experiment tracking, API deployment, and cloud deployment.
The pipeline is designed for:
* Reproducible experiments
* Scalable deployment
* Zero local infrastructure dependency

Dataset
* Source: Breast Cancer Wisconsin (Diagnostic) Data
* sklearn.datasets.load_breast_cancer()
* Features: 30 numerical features describing cell nuclei characteristics.
* Target: 0 = Malignant, 1 = Benign
Preprocessing:
* Train-test split: 80/20
* Feature scaling using StandardScaler

ML Model
* Model: LogisticRegression
* Parameters: max_iter=5000 (to ensure convergence)
* Performance:
    * Training Accuracy: ~98.9%
    * Test Accuracy: ~97.4%
* Sample prediction verified with scaled input to ensure consistency.

MLOps Pipeline
* Integrated MLflow for experiment tracking:
    * Metrics logged: train_accuracy, test_accuracy
    * Models tracked via MLflow autologging
* Model persisted locally using joblib for reliable API loading.
* Ensured feature consistency between training and inference.

FastAPI API
Endpoints:
* GET / → Health check
* POST /predict → Returns "Benign" or "Malignant"
Input validation:
* Implemented using Pydantic.
* Expects JSON input:
{
  "features": [feature1, feature2, ..., feature30]
}
Model loading:
import joblib
model = joblib.load("breast_cancer_model.pkl")

Dockerization
* Base Image: python:3.11-slim
* Two-stage build for smaller runtime image
* Dependencies installed: numpy, pandas, scikit-learn, fastapi, uvicorn, joblib
* Container testing:
docker build -t breast-cancer-api:latest .
docker run -p 8000:8000 breast-cancer-api:latest
* Local predictions verified to match Python environment outputs.
* Common issues resolved:
    * Missing files → ensured breast_cancer_model.pkl copied into /app
    * Port conflicts → free port 8000 before running container

Cloud Deployment
* Platform: Google Cloud Run
* Docker Image: Pushed to Artifact Registry
* Environment Variable: PORT dynamically assigned by Cloud Run
* Public REST endpoint:
    * GET / → Health check
    * POST /predict → JSON features → Returns prediction
Steps:
1. Build and tag Docker image locally.
2. Push image to GCP Artifact Registry.
3. Deploy to Cloud Run (unauthenticated for testing).
4. Test endpoint with Postman or curl.

Future Work
* Container orchestration with Kubernetes for scaling
* Model versioning via MLflow registry
* CI/CD integration for automated deployment
* Add user-friendly frontend for predictions
