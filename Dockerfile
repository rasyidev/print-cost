FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /src

# Copy the application into the container
COPY src/ /src
COPY pyproject.toml /src
COPY uv.lock /src
COPY main.py /src
COPY models /src/models

# Install the application dependencies
RUN uv sync --frozen --no-cache

EXPOSE 8501

# Run the application
ENTRYPOINT [".venv/bin/streamlit", "run", "main.py", "--server.port=8501", "server.address=0.0.0.0"]