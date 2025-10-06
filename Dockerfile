FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files
COPY requirements.txt .
COPY app/ ./app/
COPY model/ ./model/
COPY data/ ./data/

# Create logs directory
RUN mkdir -p logs

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Python path to ensure imports work correctly
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8000

# Use Gunicorn with Uvicorn workers
CMD ["gunicorn", \
    "app.main:app", \
    "-k", "uvicorn.workers.UvicornWorker", \
    "--workers", "4", \
    "--timeout", "60", \
    "--keep-alive", "5", \
    "--max-requests", "2000", \
    "--max-requests-jitter", "200", \
    "--bind", "0.0.0.0:8000", \
    "--worker-tmp-dir", "/dev/shm", \
    "--preload"]