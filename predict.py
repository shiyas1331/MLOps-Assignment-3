# predict.py
import joblib
from sklearn.datasets import fetch_california_housing

model = joblib.load("model.joblib")
data = fetch_california_housing()
X = data.data[:5]
predictions = model.predict(X)

print("Predictions:", predictions)
