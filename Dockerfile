FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container
COPY src/ /app
COPY pyproject.toml /app
COPY uv.lock /app
COPY main.py /app
COPY models /app/models

# Install the application dependencies
WORKDIR /app
RUN uv sync --frozen --no-cache

# Run the application
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]