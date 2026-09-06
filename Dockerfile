FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HARNESS_ROOT="/workspace/my-cluster" \
    AGENT_NAME="agent-ripdude" \
    OPENAI_BASE_URL="http://ollama-service:11434/v1" \
    MODEL_NAME=""

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps chromium

# Create unprivileged user matching default host UID (1000)
RUN useradd -m -u 1000 ripdudeuser && \
    mkdir -p /workspace/my-cluster && \
    chown -R ripdudeuser:ripdudeuser /app /workspace

COPY --chown=ripdudeuser:ripdudeuser agent.py server.py tools.py config.py ./

USER ripdudeuser

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]