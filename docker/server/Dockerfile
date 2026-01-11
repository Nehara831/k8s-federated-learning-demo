FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY docker/server/requirements.txt .
# Install with increased timeout and retries for large packages like PyTorch
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

# Copy source code
# COPY src/ ./src/
# COPY config/ ./config/
# COPY modules/ ./modules/

# Create directories for data and outputs
RUN mkdir -p /app/data
RUN mkdir -p /app/outputs
RUN mkdir -p /app/detector_output
RUN mkdir -p /app/initial_params

# Copy detector model directly into the image
COPY detector_model/ /app/detector_model/

# Pre-download DistilBERT base model during build (cached in image)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    print('Downloading DistilBERT base model...'); \
    AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
    AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2); \
    print('✓ Base model cached in image')"

CMD ["python", "-m", "src.server.server_main"]
