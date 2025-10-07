# Stage 1: Builder
FROM python:3.11 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran libopenblas-dev liblapack-dev cmake curl \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip exists
RUN python3 -m ensurepip --upgrade
RUN python3 -m pip install --upgrade pip wheel setuptools

# Copy requirements and install packages
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy app files
COPY . .

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy app code, templates, and model
COPY --from=builder /app/app ./app
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/breast_cancer_model.pkl ./breast_cancer_model.pkl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
