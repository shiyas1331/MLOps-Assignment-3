import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load original model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

weights = model.coef_
bias = model.intercept_

# Save unquantized params
unquant_params = {"weights": weights, "bias": bias}
joblib.dump(unquant_params, "unquant_params.joblib")

# Quantize weights
w_min, w_max = weights.min(), weights.max()
q_weights = np.round((weights - w_min) * 255 / (w_max - w_min)).astype(np.uint8)

# Save quantized params
quant_params = {
    "weights": q_weights,
    "bias": bias,  # keep bias unquantized
    "w_min": w_min,
    "w_max": w_max
}
joblib.dump(quant_params, "quant_params.joblib")

# Load data and scale
data = fetch_california_housing()
X = data.data
X_scaled = scaler.transform(X)
y_true = data.target

# Inference: original model
y_pred_orig = model.predict(X_scaled)
r2_orig = r2_score(y_true, y_pred_orig)

# Inference: quantized model
deq_weights = q_weights.astype(np.float32) * (w_max - w_min) / 255 + w_min
deq_bias = bias
y_pred_quant = np.dot(X_scaled, deq_weights) + deq_bias
r2_quant = r2_score(y_true, y_pred_quant)

# Get file sizes
def size_kb(path):
    return os.path.getsize(path) / 1024

size_unquant = size_kb("unquant_params.joblib")
size_quant = size_kb("quant_params.joblib")

# Print comparison table
print("\nMetric                    Original Sklearn     Quantized Model")
print("---------------------------------------------------------------")
print(f"R² Score                  {r2_orig:.3f}                 {r2_quant:.3f}")
print(f"Model Size (KB)          {size_unquant:.1f} KB           {size_quant:.1f} KB")
