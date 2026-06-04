# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (including postgresql-client for database tasks)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose default ports: Streamlit (8501) and static visual dashboard (8080)
EXPOSE 8501
EXPOSE 8080

# Environment variable defaults (can be overridden by docker-compose)
ENV DATABASE_HOST=db
ENV DATABASE_USER=postgres
ENV DATABASE_NAME=recursive_trading
ENV DATABASE_PORT=5432

# Streamlit command as default entrypoint
CMD ["streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
