from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import random
from sklearn.datasets import load_breast_cancer

app = FastAPI(title="Breast Cancer Classifier API")
model = joblib.load("breast_cancer_model.pkl")
templates = Jinja2Templates(directory="templates")

class SampleInput(BaseModel):
    features: list

# Helper to generate random examples
data = load_breast_cancer()
X = data.data
y = data.target

def get_random_examples():
    benign_idx = random.choice(np.where(y == 1)[0])
    malignant_idx = random.choice(np.where(y == 0)[0])
    benign_sample = X[benign_idx]
    malignant_sample = X[malignant_idx]
    benign_str = ",".join(map(str, benign_sample))
    malignant_str = ",".join(map(str, malignant_sample))
    return benign_str, malignant_str

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    benign_example, malignant_example = get_random_examples()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "benign_example": benign_example,
            "malignant_example": malignant_example
        }
    )

@app.post("/predict")
def predict(input_data: SampleInput):
    data = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(data)
    result = "Benign" if prediction[0] == 1 else "Malignant"
    return {"prediction": result}
