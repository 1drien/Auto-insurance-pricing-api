# 1. Image de base légère avec Python
FROM python:3.9-slim

# 2. Répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copie du fichier de dépendances
COPY requirements.txt .

# 4. Installation des dépendances (SANS CACHE pour une image légère)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie de TOUT le projet (app.py, src/, models/, data/, tests/)
COPY . .

# 6. Exposition du port 8000
EXPOSE 8000

# 7. Commande pour lancer l'API (Uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]