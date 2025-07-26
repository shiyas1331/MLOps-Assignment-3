# train.py
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")
