# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevent Python from writing .pyc files and keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements-dashboard.txt /app/
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy project files into container
COPY dashboard/ /app/dashboard/
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY results/ /app/results/
COPY outputs/ /app/outputs/
COPY reports/ /app/reports/
COPY .streamlit/ /app/.streamlit/

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to ensure container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
