FROM python:3.11-slim

WORKDIR /app

# Installer UV dans l'image Docker
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copier les fichiers de dépendances d'abord (cache Docker)
COPY pyproject.toml uv.lock ./

# Installer seulement les dépendances de prod
RUN uv sync --no-dev --frozen

# Copier le reste du projet
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]