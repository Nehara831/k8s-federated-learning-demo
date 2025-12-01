FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY docker/client/requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY modules/ ./modules/

# Create directories for data and outputs
RUN mkdir -p /app/data
RUN mkdir -p /app/outputs

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "src/client/client_main.py"]