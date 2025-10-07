import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model():
    # Load dataset
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['label'] = data.target

    X = df.drop(columns='label')
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Train model without scaling
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    joblib.dump(model, "breast_cancer_model.pkl")
    print("Model saved successfully!")

    # Sample prediction
    sample_input = np.array([
        # 0.5, 1.2, 2.3, 4.5, 1.1, 0.9, 1.3, 2.2, 3.1, 1.5, 0.7, 2.0, 3.2, 1.9, 2.5, 1.0, 1.8, 2.7, 3.3, 2.1, 1.4, 0.8, 1.6, 2.9, 3.0, 1.7, 2.6, 0.9, 1.2, 3.1
        6.5, 9.2, 12.3, 15.8, 8.9, 10.4, 13.2, 11.5, 14.6, 9.7, 
        7.3, 12.9, 16.2, 8.8, 11.9, 9.1, 13.7, 15.4, 10.8, 14.2, 
        8.5, 9.6, 11.1, 12.8, 15.1, 10.3, 14.9, 8.7, 9.9, 13.5
    ]).reshape(1, -1)

    prediction = model.predict(sample_input)
    result = "Benign" if prediction[0] == 1 else "Malignant"
    print(f"Sample Prediction: {result}")

if __name__ == "__main__":
    train_and_save_model()
