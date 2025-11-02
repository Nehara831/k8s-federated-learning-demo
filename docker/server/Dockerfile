FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY docker/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY modules/ ./modules/
COPY datasets/ ./datasets/

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "src/server/server_main.py"]