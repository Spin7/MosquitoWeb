FROM python:3.11-slim

# Install system-level GIS / image dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Ensure the models directory exists (models are downloaded at startup)
RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "Server:app", "--host", "0.0.0.0", "--port", "8000"]
