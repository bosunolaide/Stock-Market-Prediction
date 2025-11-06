# Use a lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and use unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY Model.h5 Feature_Scaler.pck Target_Scaler.pck ./
COPY app.py api.py ./

# Expose both ports
EXPOSE 8501
EXPOSE 8000

# Use supervisord or bash to run both services concurrently
# We'll use a small bash trick to run both Streamlit and FastAPI
CMD bash -c "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"
