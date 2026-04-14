# Base image with Python
FROM python:3.11-slim

# Working directory inside the container
WORKDIR /app

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (Docker cache optimization)
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --no-dev --frozen

# Copy the rest of the project (code, models, data)
COPY . .

# Document the port used by the API
EXPOSE 8000

# Start the API automatically when the container launches
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]