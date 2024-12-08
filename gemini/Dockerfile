FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies and cache the apt packages
RUN apt-get update && apt-get install -y libpq-dev gcc && apt-get clean && rm -rf /var/lib/apt/lists/*

# Separate dependencies into layers for caching
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app /app

EXPOSE 8000

CMD ["python", "app/run.py"]
