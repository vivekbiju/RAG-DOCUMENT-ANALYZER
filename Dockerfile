FROM python:3.11-slim

ENV PYTHONTOWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies + supervisor to handle managing two processes at once
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install your single root requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application assets
COPY ./src ./src
COPY ./backend ./backend
COPY ./frontend ./frontend
COPY ./available_models.py .

# --- CACHE BUST & LOG REDIRECT ---
RUN echo "Forcing clean real-time log stream tracking v3 with uploader security flags"

# --- FIXED: Added uploader security override flags to the Streamlit command line ---
RUN echo '[supervisord]\nnodaemon=true\n\n[program:backend]\ncommand=uvicorn backend.main:app --host 0.0.0.0 --port 8000\nstdout_logfile=/dev/stdout\nstdout_logfile_maxbytes=0\nstderr_logfile=/dev/stderr\nstderr_logfile_maxbytes=0\n\n[program:frontend]\ncommand=streamlit run frontend/app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false --server.enableXsrfProtection=false --server.enableCORS=false\n' > /etc/supervisor/conf.d/supervisord.conf

# Hugging Face Spaces strictly requires exposing port 7860
EXPOSE 7860

# Start Supervisor to run both the API and UI simultaneously
CMD ["/usr/bin/supervisord"]