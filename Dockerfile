# Use lightweight Python 3.10 base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy everything from current directory to container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir scikit-learn joblib

# Run training first to generate model.joblib
RUN python train.py

# Run predict.py to verify predictions
CMD ["python", "predict.py"]
