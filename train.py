from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
