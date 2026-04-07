# Base image (OS + Python)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# 5Open port for FastAPI
EXPOSE 8000

# Run your API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]