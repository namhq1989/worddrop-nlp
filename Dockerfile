FROM python:3.12.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies and Gunicorn
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Download NLTK data
RUN python -m nltk.downloader wordnet

# Download spaCy model
RUN python -m spacy download en_core_web_md

# Copy application code
COPY . .

# Expose the port the app runs on
ENV PORT=3000
EXPOSE ${PORT}

# Command to run the application in production mode with Gunicorn
CMD ["sh", "-c", "gunicorn --workers=4 --bind 0.0.0.0:${PORT} --timeout 120 app:app"]


