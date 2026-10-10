# Small official Python image (Docker uses its own Python, not your laptop's)
FROM python:3.14-slim

WORKDIR /app

# Install dependencies first, so Docker can reuse this layer when only app code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the server needs: the app code and the ONNX model
COPY app.py .
COPY models/best.onnx ./models/best.onnx

EXPOSE 8000

# 0.0.0.0 makes the server reachable from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]