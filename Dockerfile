# Use an official lightweight Python image.
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set working directory
WORKDIR /app

# Copy local code to the container image.
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup using uvicorn.
# Cloud Run expects the app to listen on the port defined by the PORT environment variable.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}