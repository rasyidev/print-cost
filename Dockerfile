FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy the application into the container
COPY src/ /app
COPY pyproject.toml /app
COPY uv.lock /app
COPY main.py /app
COPY models /app/models

# Install the application dependencies
RUN uv sync --frozen --no-cache

EXPOSE 8501

# Run the application
ENTRYPOINT [".venv/bin/streamlit", "run", "main.py", "--server.port=8501", "server.address=0.0.0.0"]