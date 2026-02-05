FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Copy the application into the container
COPY src/ /app/src
COPY pyproject.toml /app
COPY uv.lock /app
COPY main.py /app
COPY models /app/models

# Ensure Python can find packages
ENV PYTHONPATH=/app

# Install dependencies
RUN uv sync --frozen --no-cache

EXPOSE 8501

# Run the application
ENTRYPOINT [".venv/bin/streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
