# Étape de base avec Python 3.10.6
FROM python:3.10.6-slim
# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY app.py .
COPY mlruns/ mlruns/
COPY mlartifacts/ mlartifacts/
COPY tests/ tests/


# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de l'API FastAPI
EXPOSE 8000

# Lancer l'API avec uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
