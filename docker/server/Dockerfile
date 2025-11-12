FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY docker/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY modules/ ./modules/
COPY datasets/ ./datasets/

# Create directories for detector
RUN mkdir -p /app/outputs/detector
RUN mkdir -p /app/detector_model

# If you have a pre-trained detector model, copy it here:
COPY detector_model/ /app/detector_model/

CMD ["python", "-m", "src.server.server_main"]